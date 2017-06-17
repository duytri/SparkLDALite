package main.scala.obj

import org.apache.spark.mllib.linalg.{ Matrices, Matrix, Vector, Vectors }
import org.apache.spark.mllib.util.{ Loader, Saveable }
import org.apache.spark.graphx.{ Edge, EdgeContext, Graph, VertexId }
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{ Row, SparkSession }
import org.apache.hadoop.fs.Path

import breeze.linalg.{ argmax, argtopk, normalize, sum, DenseMatrix => BDM, DenseVector => BDV }
import breeze.numerics.{ exp, lgamma }

import org.json4s.DefaultFormats
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

import main.scala.helper.Utils
import main.scala.helper.BPQ

abstract class Model {

  /** Number of topics */
  def k: Int

  /** Vocabulary size (number of terms or terms in the vocabulary) */
  def vocabSize: Int

  /**
   * Concentration parameter (commonly named "alpha") for the prior placed on documents'
   * distributions over topics ("theta").
   *
   * This is the parameter to a Dirichlet distribution.
   */
  def docConcentration: Vector

  /**
   * Concentration parameter (commonly named "beta" or "eta") for the prior placed on topics'
   * distributions over terms.
   *
   * This is the parameter to a symmetric Dirichlet distribution.
   *
   * @note The topics' distributions over terms are called "beta" in the original LDA paper
   * by Blei et al., but are called "phi" in many later papers such as Asuncion et al., 2009.
   */
  def topicConcentration: Double

  /**
   * Shape parameter for random initialization of variational parameter gamma.
   * Used for variational inference for perplexity and other test-time computations.
   */
  //protected def gammaShape: Double

  /**
   * Inferred topics, where each topic is represented by a distribution over terms.
   * This is a matrix of size vocabSize x k, where each column is a topic.
   * No guarantees are given about the ordering of the topics.
   */
  def topicsMatrix: Matrix

  /**
   * Return the topics described by weighted terms.
   *
   * @param maxTermsPerTopic  Maximum number of terms to collect for each topic.
   * @param eta  Topics' distributions over terms.
   * @param termSize  Actual terms size.
   * @return  Array over topics.  Each topic is represented as a pair of matching arrays:
   *          (term indices, term weights in topic).
   *          Each topic's terms are sorted in order of decreasing weight.
   */
  def describeTopics(maxTermsPerTopic: Int): Array[Array[(Int, Double)]]

  /**
   * Return the topics described by weighted terms.
   *
   * WARNING: If vocabSize and k are large, this can return a large object!
   *
   * @return  Array over topics.  Each topic is represented as a pair of matching arrays:
   *          (term indices, term weights in topic).
   *          Each topic's terms are sorted in order of decreasing weight.
   */
  def describeTopics(): Array[Array[(Int, Double)]] = describeTopics(vocabSize)
}

/**
 * Distributed LDA model.
 * This model stores the inferred topics, the full training dataset, and the topic distributions.
 */
class LDAModel(
  val graph: Graph[LDA.TopicCounts, LDA.TokenCount],
  val globalTopicTotals: LDA.TopicCounts,
  val k: Int,
  val vocabSize: Int,
  override val docConcentration: Vector,
  override val topicConcentration: Double,
  val iterationTimes: Array[Double])
    extends Model {

  import LDA._

  /**
   * Inferred topics, where each topic is represented by a distribution over terms.
   * This is a matrix of size vocabSize x k, where each column is a topic.
   * No guarantees are given about the ordering of the topics.
   *
   * WARNING: This matrix is collected from an RDD. Beware memory usage when vocabSize, k are large.
   */
  lazy val topicsMatrix: Matrix = {
    // Collect row-major topics
    val termTopicCounts: Array[(Int, TopicCounts)] =
      graph.vertices.filter(_._1 < 0).map {
        case (termIndex, cnts) =>
          (index2term(termIndex), cnts)
      }.collect()
    // Convert to Matrix
    val brzTopics = BDM.zeros[Double](vocabSize, k)
    termTopicCounts.foreach {
      case (term, cnts) =>
        var j = 0
        while (j < k) {
          brzTopics(term, j) = cnts(j)
          j += 1
        }
    }
    Utils.matrixFromBreeze(brzTopics)
  }
  
  override def describeTopics(maxTermsPerTopic: Int): Array[Array[(Int, Double)]] = {
    val numTopics = k
    val phi = computePhi
    val result = Array.ofDim[(Int, Double)](k, maxTermsPerTopic)
    for (topic <- 0 until k) {
      val maxVertexPerTopic = phi.filter(_._1 == topic).takeOrdered(maxTermsPerTopic)(Ordering[Double].reverse.on(_._3))
      result(topic) = maxVertexPerTopic.map {
        case (topicId, termId, phi) =>
          (index2term(termId), phi)
      }
    }
    return result
  }

  def computeTheta(): RDD[(VertexId, Int, Double)] = {
    val alpha = this.docConcentration(0)
    graph.vertices.filter(LDA.isDocumentVertex).flatMap {
      case (docId, topicCounts) =>
        topicCounts.mapPairs {
          case (topicId, wordCounts) =>
            val thetaMK = ((wordCounts + alpha) / (topicCounts.data.sum + topicCounts.length * alpha))
            (docId, topicId, thetaMK)
        }.toArray
    }
  }

  def computePhi(): RDD[(Int, VertexId, Double)] = {
    val eta = this.topicConcentration
    val wordTopicCounts = this.globalTopicTotals
    val vocabSize = this.vocabSize
    graph.vertices.filter(LDA.isTermVertex).flatMap {
      case (termId, topicCounts) =>
        topicCounts.mapPairs {
          case (topicId, wordCounts) =>
            val phiKW = ((wordCounts + eta) / (wordTopicCounts.data(topicId) + vocabSize * eta))
            (topicId, termId, phiKW)
        }.toArray
    }
  }

  def computePerplexity(tokenSize: Long): Double = {
    val alpha = this.docConcentration(0) // To avoid closure capture of enclosing object
    val eta = this.topicConcentration
    val N_k = globalTopicTotals
    val smoothed_N_k: TopicCounts = N_k + (vocabSize * eta)
    // Edges: Compute token log probability from phi_{wk}, theta_{kj}.
    val sendMsg: EdgeContext[TopicCounts, TokenCount, Double] => Unit = (edgeContext) => {
      val N_wj = edgeContext.attr
      val smoothed_N_wk: TopicCounts = edgeContext.dstAttr + eta - 1.0
      val smoothed_N_kj: TopicCounts = edgeContext.srcAttr + alpha - 1.0
      val phi_wk: TopicCounts = smoothed_N_wk :/ smoothed_N_k
      val theta_kj: TopicCounts = normalize(smoothed_N_kj, 1.0)
      val tokenLogLikelihood = N_wj * math.log(phi_wk.dot(theta_kj))
      edgeContext.sendToDst(tokenLogLikelihood)
    }
    val docSum = graph.aggregateMessages[Double](sendMsg, _ + _)
      .map(_._2).fold(0.0)(_ + _)
    return math.exp(-1 * docSum / tokenSize)
  }

  def countGraphInfo() = {
    println("Number of document vertices: " + graph.vertices.filter(LDA.isDocumentVertex).count())
    println("Number of term vertices: " + graph.vertices.filter(LDA.isTermVertex).count())
    println("Number of edges: " + graph.edges.count())
  }
}
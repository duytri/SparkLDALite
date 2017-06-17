package main.scala.helper

import org.apache.spark.graphx._
import main.scala.obj.LDA.{ TopicCounts, TokenCount }
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{ DenseVector, Matrices, SparseVector, Vector, Vectors }
import main.scala.obj.LDA
import breeze.linalg.{ all, normalize, sum, DenseMatrix => BDM, DenseVector => BDV }
import scala.util.Random
import main.scala.obj.LDAModel
import main.scala.obj.Model

class LDAOptimizer {
  import LDA._

  // The following fields will only be initialized through the initialize() method
  var graph: Graph[TopicCounts, TokenCount] = null
  var k: Int = 0
  var vocabSize: Int = 0
  var docConcentration: Double = 0
  var topicConcentration: Double = 0

  /**
   * Compute bipartite term/doc graph.
   */
  def initialize(
    docs: RDD[(Long, Vector)],
    vocabSize: Long,
    lda: LDA): LDAOptimizer = {
    // LDAOptimizer currently only supports symmetric document-topic priors
    val docConcentration = lda.getDocConcentration

    val topicConcentration = lda.getTopicConcentration
    val k = lda.getK

    this.docConcentration = if (docConcentration <= 0) 50.0 / k else docConcentration
    this.topicConcentration = if (topicConcentration <= 0) 1.1 else topicConcentration
    val randomSeed = lda.getSeed

    // For each document, create an edge (Document -> Term) for each unique term in the document.
    val edges: RDD[Edge[TokenCount]] = docs.flatMap {
      case (docID: Long, termCounts: Vector) =>
        // Add edges for terms with non-zero counts.
        Utils.asBreeze(termCounts).activeIterator.filter(_._2 != 0.0).map {
          case (term, cnt) =>
            Edge(docID, term2index(term), cnt)
        }
    }

    // Create vertices.
    // Initially, we use random soft assignments of tokens to topics (random gamma).
    val docTermVertices: RDD[(VertexId, TopicCounts)] = {
      val verticesTMP: RDD[(VertexId, TopicCounts)] =
        edges.flatMap { edge =>
          val gamma = Utils.randomVectorInt(k, edge.attr.toInt)
          Seq((edge.srcId, gamma), (edge.dstId, gamma))
        }
      //val docVertices = verticesTMP.filter(LDA.isDocumentVertex).reduceByKey(_ + _) //.reduceByKey((a, b) => a)
      //val termVertices = verticesTMP.filter(LDA.isTermVertex)
      //docVertices ++ termVertices
      verticesTMP
    }

    // Partition such that edges are grouped by document
    this.graph = Graph(docTermVertices, edges).partitionBy(PartitionStrategy.EdgePartition1D)
    this.k = k
    this.vocabSize = docs.take(1).head._2.size
    this.globalTopicTotals = computeGlobalTopicTotals()
    this
  }

  def next(): LDAOptimizer = {
    require(graph != null, "graph is null, EMLDAOptimizer not initialized.")

    val eta = topicConcentration
    val W = vocabSize
    val alpha = docConcentration

    val N_k = globalTopicTotals
    val sendMsg: EdgeContext[TopicCounts, TokenCount, (Boolean, TopicCounts)] => Unit =
      (edgeContext) => {
        // Compute N_{wj} gamma_{wjk}
        val N_wj = edgeContext.attr
        // E-STEP: Compute gamma_{wjk} (smoothed topic distributions), scaled by token count
        // N_{wj}.
        val scaledTopicDistribution: TopicCounts =
          computePTopic(edgeContext.srcAttr, edgeContext.dstAttr, N_k, W, eta, alpha)
        edgeContext.sendToDst((false, edgeContext.dstAttr + scaledTopicDistribution))
        edgeContext.sendToSrc((false, edgeContext.srcAttr + scaledTopicDistribution))
      }
    // The Boolean is a hack to detect whether we could modify the values in-place.
    // TODO: Add zero/seqOp/combOp option to aggregateMessages. (SPARK-5438)
    val mergeMsg: ((Boolean, TopicCounts), (Boolean, TopicCounts)) => (Boolean, TopicCounts) =
      (m0, m1) => {
        val sum =
          if (m0._1) {
            m0._2 += m1._2
          } else if (m1._1) {
            m1._2 += m0._2
          } else {
            m1._2 + m0._2
          }
        (true, sum)
      }
    // M-STEP: Aggregation computes new N_{kj}, N_{wk} counts.
    val docTopicDistributions: VertexRDD[TopicCounts] =
      graph.aggregateMessages[(Boolean, TopicCounts)](sendMsg, mergeMsg)
        .mapValues(_._2)
    // Update the vertex descriptors with the new counts.
    graph =  Graph(docTopicDistributions, graph.edges)
    globalTopicTotals = computeGlobalTopicTotals()
    this
  }

  /**
   * Aggregate distributions over topics from all term vertices.
   *
   * Note: This executes an action on the graph RDDs.
   */
  var globalTopicTotals: TopicCounts = null

  private def computeGlobalTopicTotals(): TopicCounts = {
    val numTopics = k
    graph.vertices.filter(isDocumentVertex).values.fold(BDV.zeros[Double](numTopics))(_ += _)
  }

  def getLDAModel(iterationTimes: Array[Double]): LDAModel = {
    require(graph != null, "graph is null, LDAOptimizer not initialized.")
    // The constructor's default arguments assume gammaShape = 100 to ensure equivalence in
    // LDAModel.toLocal conversion.
    new LDAModel(this.graph, this.globalTopicTotals, this.k, this.vocabSize,
      Vectors.dense(Array.fill(this.k)(this.docConcentration)), this.topicConcentration,
      iterationTimes)
  }
}
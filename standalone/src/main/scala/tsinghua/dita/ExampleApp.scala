package tsinghua.dita

import java.io.{FileOutputStream, ObjectInputStream, ObjectOutputStream}

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FSDataInputStream, FSDataOutputStream, FileSystem, Path}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel
import tsinghua.dita.algorithms.{TrajectoryRangeAlgorithms, TrajectorySimilarityWithKNNAlgorithms, TrajectorySimilarityWithThresholdAlgorithms}
import tsinghua.dita.common.DITAConfigConstants
import tsinghua.dita.common.shape.{Point, Rectangle}
import tsinghua.dita.common.trajectory.{Trajectory, TrajectorySimilarity}
import tsinghua.dita.index.LocalIndex
import tsinghua.dita.index.global.GlobalTrieIndex
import tsinghua.dita.index.local.LocalTrieIndex
import tsinghua.dita.partition.PackedPartition
import tsinghua.dita.partition.global.GlobalTriePartitioner
import tsinghua.dita.rdd.TrieRDD

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
object ExampleApp {
  Logger.getLogger("org").setLevel(Level.ERROR)

  def testJoin(args:Array[String]):Unit = {
    val conf = new SparkConf().setAppName("DITADistanceJoin")
    val sc = new SparkContext(conf)
    val hconf = sc.hadoopConfiguration
    hconf.setInt("dfs.replication", 1)
    val traj_file = args(0)
    val threshold = args(1).toDouble
    val rate = args(2).toDouble
    val partition = args(3).toInt
    DITAConfigConstants.GLOBAL_NUM_PARTITIONS = partition
    val output_file = args(4)
    Console.setOut(new FileOutputStream(output_file))
//    val (partitionedRDD, partitioner) = GlobalTriePartitioner.partition(trajs)
    val trajs = sc.objectFile[String](traj_file).sample(false,rate,System.currentTimeMillis()).zipWithIndex().map(getTrajectory)
    val rdd1 = new TrieRDD(trajs)
    val rdd2 = new TrieRDD(trajs)
    val start = System.nanoTime()
    val thresholdJoin = TrajectorySimilarityWithThresholdAlgorithms.FineGrainedDistributedJoin
    val thresholdJoinAnswer = thresholdJoin.join(sc, rdd1, rdd2, TrajectorySimilarity.DTWDistance, threshold).count()
    val end = System.nanoTime()
    println(s"res number is: ${thresholdJoinAnswer}")
    println(s"batch search cost is: ${(end-start)*1.0/1000000}ms")
    rdd1.packedRDD.unpersist()
    rdd2.packedRDD.unpersist()
    sc.stop()
  }
  def saveGlobalIndex(globalIndex:GlobalTrieIndex,output:String) = {
    val hdfs = FileSystem.get(new Configuration())
    val path = new Path(output)
    val oos = new ObjectOutputStream(new FSDataOutputStream(hdfs.create(path)))
    oos.writeObject(globalIndex)
    oos.close
//    hdfs.close()
  }
  def loadGlobalIndex(input:String) = {
    val hdfs = FileSystem.get(new Configuration())
    val path = new Path(input)
    val ois = new ObjectInputStream(new FSDataInputStream(hdfs.open(path)))
    val sample_model = ois.readObject.asInstanceOf[GlobalTrieIndex]
//    ois.close()
//    hdfs.close()
    sample_model
  }

  def generateRDD(args:Array[String]):Unit = {
    val conf = new SparkConf().setAppName("DITAGenerateRDD")
    val sc = new SparkContext(conf)
    val hconf =sc.hadoopConfiguration
    hconf.setInt("dfs:replication",1)
    val traj_file = args(0)
    val rdd1_out_put = args(1)
    val index_out_put = args(2)
    val rate = args(3).toDouble
    val partition = args(4).toInt
    DITAConfigConstants.GLOBAL_NUM_PARTITIONS = partition
    val trajs = sc.objectFile[String](traj_file).sample(false,rate,System.currentTimeMillis()).zipWithIndex().map(getTrajectory)
//    val hdfs = FileSystem.get(new Configuration())
//    hdfs
    val (partitionedRDD, partitioner) = GlobalTriePartitioner.partition(trajs)
    val rdd1 = partitionedRDD.mapPartitionsWithIndex { case (index, iter) =>
      val data = iter.toArray
      val indexes = ArrayBuffer.empty[LocalIndex]
      indexes.append(LocalTrieIndex.buildIndex(data))
      Array(PackedPartition(index, data, indexes.toArray)).iterator
    }
    val index = GlobalTrieIndex(partitioner)
    rdd1.saveAsObjectFile(rdd1_out_put)
    saveGlobalIndex(index, index_out_put)
    sc.stop()
  }
  def testJoin_2(args:Array[String]):Unit = {
    val conf = new SparkConf().setAppName("DITADistanceJoin")
    val sc = new SparkContext(conf)
    val hconf = sc.hadoopConfiguration
    hconf.setInt("dfs.replication", 1)
    val rdd1_output = args(0)
    val index_output = args(1)
//    val traj_file = args(0)
//    val trajs = sc.objectFile[String](traj_file).zipWithIndex().map(getTrajectory)
    val threshold = args(2).toDouble
    val output_file = args(3)
    Console.setOut(new FileOutputStream(output_file))
    //    val (partitionedRDD, partitioner) = GlobalTriePartitioner.partition(trajs)
//    val rdd1 = new TrieRDD(trajs)
//    val rdd2 = rdd1
    val rdd1 = sc.objectFile[PackedPartition](rdd1_output)
    val index_1 = loadGlobalIndex(index_output)
    rdd1.persist(StorageLevel.MEMORY_AND_DISK_SER)
    val rdd2 = sc.objectFile[PackedPartition](rdd1_output)
    rdd2.persist(StorageLevel.MEMORY_AND_DISK_SER)
    val index_2 = loadGlobalIndex(index_output)
    rdd1.count()
    rdd2.count()
    val trieRDD1 = new TrieRDD(sc.emptyRDD[Trajectory])
    val trieRDD2 = new TrieRDD(sc.emptyRDD[Trajectory])
    trieRDD1.update(rdd1,index_1)
    trieRDD2.update(rdd2,index_2)
    val start = System.nanoTime()
    val thresholdJoin = TrajectorySimilarityWithThresholdAlgorithms.FineGrainedDistributedJoin
    val thresholdJoinAnswer = thresholdJoin.join(sc, trieRDD1, trieRDD2, TrajectorySimilarity.DTWDistance, threshold).count()
    val end = System.nanoTime()
    rdd1.unpersist()
    rdd2.unpersist()
    println(s"res number is: ${thresholdJoinAnswer}")
    println(s"batch search cost is: ${(end-start)*1.0/1000000}ms")
    sc.stop()
  }

  def batchSearch(args:Array[String]):Unit = {
    val conf = new SparkConf().setAppName("DITABatchQuery")
    val sc = new SparkContext(conf)
    val hconf = sc.hadoopConfiguration
    hconf.setInt("dfs.replication", 1)
    val traj_file = args(0)

    val query_number = args(1).toInt
    val threshold = args(2).toDouble
    val partition = args(3).toInt
    val output_file = args(4)
    DITAConfigConstants.GLOBAL_NUM_PARTITIONS = partition
    val rdd1_output = if(args.length>=6) args(5) else null
    val index_output = if(args.length>=7) args(6) else null
    val query_output = if(args.length>=8) args(7) else null
    val hdfs = FileSystem.get(new Configuration())
    Console.setOut(new FileOutputStream(output_file))
    val(rdd1,index,queryTrajectory) =  if(!hdfs.exists(new Path(rdd1_output))) {
      val trajs = sc.objectFile[String](traj_file).zipWithIndex().map(getTrajectory)
      val hdfs = FileSystem.get(new Configuration())
      val queryTrajectory = if(!hdfs.exists(new Path(query_output))) {
       val tt =  trajs.take(query_number)
        val queryRdd = sc.parallelize(tt, 1)
        queryRdd.saveAsObjectFile(query_output)
        tt
      }else{sc.objectFile[Trajectory](query_output).collect()}
      //sample(false,query_number*1.0/5000000,System.currentTimeMillis()).collect();
      val (partitionedRDD, partitioner) = GlobalTriePartitioner.partition(trajs)
      val rdd1 = partitionedRDD.mapPartitionsWithIndex { case (index, iter) =>
        val data = iter.toArray
        val indexes = ArrayBuffer.empty[LocalIndex]
        indexes.append(LocalTrieIndex.buildIndex(data))
        Array(PackedPartition(index, data, indexes.toArray)).iterator
      }
      val index = GlobalTrieIndex(partitioner)
      rdd1.saveAsObjectFile(rdd1_output)
      saveGlobalIndex(index, index_output)
      (rdd1,index,queryTrajectory)
    } else{
      (sc.objectFile[PackedPartition](rdd1_output),loadGlobalIndex(index_output),if(!hdfs.exists(new Path(query_output))) {
        val trajs = sc.objectFile[String](traj_file).zipWithIndex().map(getTrajectory)
        val tt =  trajs.take(query_number)
        val queryRdd = sc.parallelize(tt, 1)
        queryRdd.saveAsObjectFile(query_output)
        tt
      }else{sc.objectFile[Trajectory](query_output).collect()})
    }


    rdd1.persist(StorageLevel.MEMORY_AND_DISK_SER)
    println(s">>>>>>>>>>>>>${rdd1.count()}")
    println("--------------------------query process-------------------")
    val start = System.nanoTime()
    val thresholdSearch = TrajectorySimilarityWithThresholdAlgorithms.DistributedSearch
    val res = thresholdSearch.batchSearch(sc,queryTrajectory,rdd1,index,TrajectorySimilarity.DTWDistance, threshold)
    val end = System.nanoTime()
    println(s"res number is: ${res}")//.length}")
    println(s"batch search cost is: ${(end-start)*1.0/(1000*query_number)/1000}ms")
    rdd1.unpersist()
    sc.stop()
  }

  def singleSearch(args:Array[String]):Unit = {
//      val spark = SparkSession
//        .builder()
//        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
//        .getOrCreate()
//      val sc = spark.sparkContext
val conf = new SparkConf().setAppName("DITAQuery")//.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer").set("spark.kryo.registrationRequired","true")
//    conf.registerKryoClasses(Array(Trajectory.getClass,PackedPartition.getClass))
    val sc = new SparkContext(conf)
    val hconf = sc.hadoopConfiguration
    hconf.setInt("dfs.replication", 1)
      val traj_file = args(0)
      val trajs  = sc.objectFile[String](traj_file).zipWithIndex().map(getTrajectory)


    val query_number = args(1).toInt
    val threshold = args(2).toDouble
    val output_file = args(3)
    Console.setOut(new FileOutputStream(output_file))
    println(s"Trejectory count:${trajs.count()}")
//    println(trajs.map(_.points.length).collect().mkString("--"))
    val queryTrajectory = trajs.take(query_number)
//    val trie = new TrieRDD(trajs)
    val (partitionedRDD, partitioner) = GlobalTriePartitioner.partition(trajs)
    val rdd1 = partitionedRDD.mapPartitionsWithIndex { case (index, iter) =>
      val data = iter.toArray
      val indexes = ArrayBuffer.empty[LocalIndex]
      indexes.append(LocalTrieIndex.buildIndex(data))
      Array(PackedPartition(index, data, indexes.toArray)).iterator
    }

//    val rdd1 = trie.packedRDD
    val index = GlobalTrieIndex(partitioner)//trie.globalIndex.asInstanceOf[GlobalTrieIndex]
    println("---------------------data distribution----------------")
    rdd1.cache()
    println(s">>>>>>>>>>>>>${rdd1.count()}")
    val gct = rdd1.map(_.data.size).collect().sum
    println(s"Global RDD partitions.length: ${rdd1.partitions.length}")
    println(s"Global trajectory count:${gct}")
    println(s"Query number: ${queryTrajectory.length}")
    val data_sitribution = rdd1.mapPartitionsWithIndex(
      (id,it) =>{
        if(!it.hasNext) {
          Iterator()
        } else {
          val p  = it.next()
          Iterator((id,p.data.length))
        }
      }
    ).collect()
    data_sitribution.foreach(x=>print(x._1+" "+x._2+";;; "))
    println()
    //////////////////////////
    //distance query
    println("--------------------------query process-------------------")
    val global_toal_cost = new mutable.ArrayBuffer[Double]()
    val global_local_prune_cost = new mutable.ArrayBuffer[Double]()
    val global_prune_parser = new mutable.ArrayBuffer[Double]()
    val global_prune_num = new mutable.ArrayBuffer[Double]()
    val local_prune_num = new mutable.ArrayBuffer[Double]()
    val thresholdSearch = TrajectorySimilarityWithThresholdAlgorithms.DistributedSearch
    queryTrajectory.foreach({
      trajectory =>
    val res = thresholdSearch.search_1(sc, trajectory, rdd1,index, TrajectorySimilarity.DTWDistance, threshold)
        val res_arr = res._1
        println("############# global query results ################")
        println("Results Num: "+ res_arr.length)
        /////////////////////////
        //global effeciency
        println("############## global and local query effeciency #####################")
        println("Global All Costs: " + res._2(0))
        println("Global Pruned Costs: " + res._2(1))
        println("Result Collect Costs: "+res._2(2))
        //     println("Global Partition Pruned Num: " + (global_rdd.partitions.length-res._3(3)))
        //     println("Global Partition Pruned Rate: "+((global_rdd.partitions.length-res._3(3))*1.0/global_rdd.partitions.length))
        println(s"Global Prune Number ratio: ${(gct-res._2(3))*1.0/gct}; prune Number is ${gct-res._2(3)}")
        println(s"Local Prune Number ratio: ${(res._2(3)-res._2(4))*1.0/res._2(3)}; prune Number is ${res._2(3)-res._2(4)}")
          global_toal_cost.append(res._2(0))
          global_local_prune_cost.append(res._2(0) - res._2(2))
          global_prune_parser.append(res._2(1))
          global_prune_num.append(res._2(3).toLong)
          local_prune_num.append(res._2(4).toLong)

    })
    val global_first_arr = global_toal_cost.toArray
    val global_second_arr = global_local_prune_cost.toArray
    val number_test = global_first_arr.length
    println(s">>>>>>> Test Numbe: ${number_test}")
    println(s">>>>>>  Global Total Cost=> Max:${global_first_arr.max}ms  Min:${global_first_arr.min}ms  Average:${global_first_arr.sum/global_first_arr.length}ms")
    println(s">>>>>>  Global And Local Prunue & Refine Cost=> Max:${global_second_arr.max}ms  Min:${global_second_arr.min}ms  Average:${global_second_arr.sum/global_second_arr.length}ms")
    println(s">>>>>> Global Prune Cost=> Max:${global_prune_parser.max}ms  Min:${global_prune_parser.min}ms  Average: ${global_prune_parser.sum/global_prune_parser.length}ms")
    println(s">>>>>>> Global Prune ratio => Average:${(number_test*gct.toLong-global_prune_num.sum)*1.0/(number_test*gct.toLong)*100}%; Prune Number => Average:${(number_test*gct.toLong-global_prune_num.sum)*1.0/number_test}")
    println(s">>>>>>>> Local Prune ratio => Average:${(global_prune_num.sum-local_prune_num.sum)*1.0/global_prune_num.sum*100}%; Prune Number => Average:${(global_prune_num.sum-local_prune_num.sum)*1.0/number_test}")
    rdd1.unpersist()
    sc.stop()
      }


  private def getTrajectory(line: (String, Long)): Trajectory = {
    val points = line._1.split(";").map(_.split(","))
      .map(x => Point(x.map(_.toDouble)))
    Trajectory(points)
  }
  def main(args:Array[String]) = {
//      singleSearch(args)
    batchSearch(args)
//    testJoin(args)
//    generateRDD(args)
//    testJoin_2(args)
  }

//  def rawTest(args: Array[String]): Unit = {
//    val spark = SparkSession
//      .builder()
//      .master("local[*]")
//      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
//      .getOrCreate()
//
//    val trajs = spark.sparkContext
//      .textFile("src/main/resources/trajectory.txt")
//      .zipWithIndex().map(getTrajectory)
//      .filter(_.points.length >= DITAConfigConstants.TRAJECTORY_MIN_LENGTH)
//      .filter(_.points.length <= DITAConfigConstants.TRAJECTORY_MAX_LENGTH)
//    println(s"Trajectory count: ${trajs.count()}")
//
//    val rdd1 = new TrieRDD(trajs)
//    val rdd2 = new TrieRDD(trajs)
//
//    // threshold-based search
//    val queryTrajectory = trajs.take(1).head
//    val thresholdSearch = TrajectorySimilarityWithThresholdAlgorithms.DistributedSearch
//    val thresholdSearchAnswer = thresholdSearch.search(spark.sparkContext, queryTrajectory, rdd1, TrajectorySimilarity.DTWDistance, 0.005)
//    println(s"Threshold search answer count: ${thresholdSearchAnswer.count()}")
//
//
//    // knn search
//    val knnSearch = TrajectorySimilarityWithKNNAlgorithms.DistributedSearch
//    val knnSearchAnswer = knnSearch.search(spark.sparkContext, queryTrajectory, rdd1, TrajectorySimilarity.DTWDistance, 100)
//    println(s"KNN search answer count: ${knnSearchAnswer.count()}")
//
//    // threshold-based join
//    val thresholdJoin = TrajectorySimilarityWithThresholdAlgorithms.FineGrainedDistributedJoin
//    val thresholdJoinAnswer = thresholdJoin.join(spark.sparkContext, rdd1, rdd2, TrajectorySimilarity.DTWDistance, 0.005)
//    println(s"Threshold join answer count: ${thresholdJoinAnswer.count()}")
//
//    // threshold-based search
//    val knnJoin = TrajectorySimilarityWithKNNAlgorithms.DistributedJoin
//    val knnJoinAnswer = knnJoin.join(spark.sparkContext, rdd1, rdd2, TrajectorySimilarity.DTWDistance, 100)
//    println(s"KNN join answer count: ${knnJoinAnswer.count()}")
//
//    // mbr range search
//    val search = TrajectoryRangeAlgorithms.DistributedSearch
//    val mbr = Rectangle(Point(Array(39.8, 116.2)), Point(Array(40.0, 116.4)))
//    val mbrAnswer = search.search(spark.sparkContext, mbr, rdd1, 0.0)
//    println(s"MBR range search count: ${mbrAnswer.count()}")
//
//    // circle range search
//    val center = Point(Array(39.9, 116.3))
//    val radius = 0.1
//    val circleAnswer = search.search(spark.sparkContext, center, rdd1, radius)
//    println(s"Circle range search count: ${circleAnswer.count()}")
//  }
}

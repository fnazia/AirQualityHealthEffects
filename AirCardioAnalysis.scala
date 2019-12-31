import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{ IndexToString, StringIndexer, VectorIndexer }
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{ ParamGridBuilder, CrossValidator }
import org.apache.spark.ml.{ Pipeline, PipelineStage }
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.tree.{Node, InternalNode, LeafNode, Split, CategoricalSplit, ContinuousSplit}
import scala.collection.mutable.Builder
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}


object AirCardioAnalysis {
  case class Air (
    mortalityclass: Double,
    gooddays: Double, moderatedays: Double, unhealthysensitivedays: Double, unhealthydays: Double,
    veryunhealthydays: Double, hazardousdays: Double, maxAQI: Double, ninetypercentileAQI: Double, medianAQI: Double,
    codays: Double, no2days: Double, ozonedays: Double, so2days: Double, pm25days: Double,
    pm10days: Double
  )
  
  def parseAir(line: Array[String]): Air = {
    Air (
      line(41).toDouble,
      line(5).toDouble, line(6).toDouble, line(7).toDouble, line(8).toDouble, line(9).toDouble,
      line(10).toDouble, line(11).toDouble, line(12).toDouble, line(13).toDouble, line(14).toDouble,
      line(15).toDouble, line(16).toDouble, line(17).toDouble, line(18).toDouble, line(19).toDouble
    )
  }
  
  def parseRDD(rdd: RDD[String]): RDD[Array[String]] = {
    //rdd.map(_.split(",")).map(_.map(_.toString))
    rdd.map(_.split(","))
  }
  
  def parseTree(
   node: Node, 
   listBuffer: Builder[(Int, Double, Double),List[(Int, Double, Double)]]
  ): List[(Int, Double, Double)] = {
  
   // splits only occur at internal nodes
    if (node.isInstanceOf[InternalNode]) {
        val inode = node.asInstanceOf[InternalNode]
 
        // only interested in continuous feature splits
        if (inode.split.isInstanceOf[ContinuousSplit]) {
 
         // extract data from split
            val split = inode.split.asInstanceOf[ContinuousSplit]
            val infoGain = inode.gain
            val splitFeature = split.featureIndex
            val splitValue = split.threshold
 
            // append data to buffer
            listBuffer += ((splitFeature, splitValue, infoGain))
        }
 
        // recursively parse the left and right child nodes
        parseTree(inode.leftChild, listBuffer)
        parseTree(inode.rightChild, listBuffer)
    }
 
    // return results
    listBuffer.result
  }
  
     
  def main(args: Array[String]): Unit={

    val conf = new SparkConf().setAppName("airqcardio").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    import sqlContext._
    import sqlContext.implicits._
    
    //load input csv file
    val rdd = sc.textFile(args(0))
    
    //get rid of header row
    val newrdd = rdd.mapPartitionsWithIndex {
      (idx, iter) => if (idx == 0) iter.drop(1) else iter 
    }
    
    //extract required attributes for feature vector and convert those into a dataframe
    val airDF = parseRDD(newrdd).map(parseAir).toDF().cache()
    airDF.registerTempTable("air")
    airDF.printSchema
    
    /** //print average-good-days, average-unhealthy-days, average-max-aqi for each mortality-class--
    sqlContext.sql("SELECT mortalityclass, avg(gooddays) as avggood, avg(unhealthydays) as avgunhealthy, avg(maxAQI) as avgmaxaqi  FROM air GROUP BY mortalityclass").show**/
    
    /** //save these average values in a text file if needed--
    val classavg = sqlContext.sql("SELECT mortalityclass, avg(gooddays) as avggood, avg(unhealthydays) as avgunhealthy, avg(maxAQI) as avgmaxaqi  FROM air GROUP BY mortalityclass").rdd
    classavg.saveAsTextFile(args(3))**/
    
    /** //print summary (count, average, min, max, stddev etc.) of the values in column 'gooddays'
    airDF.describe("gooddays").show**/
    
    /** //print average of a specific column (e.g., gooddays) for each mortality-class
    airDF.groupBy("mortalityclass").avg("gooddays").show**/
    
    //create an array of those column names that will be used for creating feature vector
    
    val featureCols = Array("gooddays", "moderatedays", "unhealthysensitivedays", "unhealthydays", "veryunhealthydays",
      "hazardousdays", "maxAQI", "ninetypercentileAQI", "medianAQI", "codays",
      "no2days", "ozonedays", "so2days", "pm25days", "pm10days"
    )
    
    //create the feature vector
    
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val df2 = assembler.transform(airDF)
    
    //try VectorIndexer
    /**val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(2)
    val df2 = featureIndexer.fit(df22).transform(df22)**/
    
    
    /** //print the feature-class table along with the feature vector
    df2.show**/
    
    //create the label column from the mortalityclass column.
    //note that StringIndexer does labeling in descending order of mortalityclass-occurance-frequency
    //e.g., if class 1 occurs most, its label will be 0 and so on. 
    
    val labelIndexer = new StringIndexer().setInputCol("mortalityclass").setOutputCol("label")
    val df3 = labelIndexer.fit(df2).transform(df2)
    
    /** //print the table with label
    df3.show**/
    
    val splitSeed = 5043
    
    //random split of total input dataset into train and test sets
    val Array(trainingData, testData) = df3.randomSplit(Array(0.8, 0.2), splitSeed)
    
    //count the number of samples for each class
    val numNegatives = trainingData.filter(trainingData("label") === 0).count
    val numPositives = trainingData.filter(trainingData("label") === 1).count
    val traindatacount = trainingData.count
    val fraction = numPositives/traindatacount
    println(traindatacount)
    
    val decrease = trainingData.filter(trainingData("label") === 0)
    val increase = trainingData.filter(trainingData("label") === 1)
    val sampleratio = increase.count().toDouble / trainingData.count().toDouble
    val decreasesubset = decrease.sample(true, sampleratio)
    val dd = increase.unionAll(decreasesubset)

    //Training without cross-validator/paramgrid/pipeline
    
    //define the classifier with its hyperparameters. Tune these values to get better accuracy
    /**val classifier = new RandomForestClassifier().setImpurity("entropy").setMinInstancesPerNode(1).setMaxDepth(10).setNumTrees(100).setFeatureSubsetStrategy("auto").setSeed(5043)
    
    //train the data with the defined classifier and create the model
    //val model = classifier.fit(trainingData)
    
    //time starts
    val startTime = System.nanoTime() 
    
    //train with balanced data
    val model = classifier.fit(dd)
    
    //training time
    val elapsedTime = (System.nanoTime() - startTime) / 1e9
    val trtime = "Training time: " +  elapsedTime + " seconds"

    //define an evaluator to measure the model performance (e.g., accuracy, AUROC etc.)
    val evaluator = new BinaryClassificationEvaluator().setLabelCol("label")
    
    //test the model on test data
    val predictions = model.transform(testData)
    model.toDebugString

    //get the features' ranking
    val importances = model.featureImportances
    
    //save feature ranking with respective name
    val res = featureCols.zip(importances.toArray).sortBy(-_._2)
    val imf = sc.parallelize(res)
    imf.saveAsTextFile(args(1))
    
    
    //get the accuracy from the trained model
    val accuracy = evaluator.evaluate(predictions)
    val rawacc = "Accuracy\n" + accuracy
    //println("accuracy before pipeline fitting " + accuracy)
    
    
    //see the hyperparameters key-value
    val modelparam = "model parameters:\n" + model.extractParamMap()
    
    //print performance metrics
    val labelAndPreds = predictions.select("prediction", "label").rdd.map(x =>
        (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))
    
    val metrics = new BinaryClassificationMetrics(labelAndPreds)
    val cm = new MulticlassMetrics(labelAndPreds)
    
    val auc = "AUC\n" + metrics.areaUnderROC()
    val confmat = "Confusion Matrix\n" + cm.confusionMatrix
    
    //println(metrics.areaUnderROC())
    //println(cm.confusionMatrix)
    
    
    // Precision by label
    val mlabels = cm.labels
    val f = mlabels.array
    
    val pre1 = "Precision(" + f(0).toInt + ")" + " = " + cm.precision(f(0))
    val pre2 = "Precision(" + f(1).toInt + ")" + " = " + cm.precision(f(1))
    
    //Recall by label
    val rec1 = "Recall(" + f(0).toInt + ")" + " = " + cm.recall(f(0))
    val rec2 = "Recall(" + f(1).toInt + ")" + " = " + cm.recall(f(1))
    
    //False positive rate by label
    val fpr1 = "FPR(" + f(0).toInt + ")" + " = " + cm.falsePositiveRate(f(0))
    val fpr2 = "FPR(" + f(1).toInt + ")" + " = " + cm.falsePositiveRate(f(1))
    
    //F-measure by label
    val fmes1 = "F1-Score(" + f(0).toInt + ")" + " = " + cm.fMeasure(f(0))
    val fmes2 = "F1-Score(" + f(1).toInt + ")" + " = " + cm.fMeasure(f(1))
    
    //get pointer to the model 
    val rfModel = model
     
    //parameters required for plots
     
     val trainlen = dd.count()
     val trainsize = List.range(10, trainlen, 100)
     var train_acc = List[Double]()
     var valid_acc = List[Double]()
     
     for (i <- trainsize) {
       val plotmodel = classifier.fit(dd.limit(i.toInt))
       val train_prediction = plotmodel.transform(dd.limit(i.toInt))
       val train_accuracy = evaluator.evaluate(train_prediction)
       train_acc = train_acc :+ train_accuracy
       
       val valid_prediction = plotmodel.transform(testData)
       val valid_accuracy = evaluator.evaluate(valid_prediction)
       valid_acc = valid_acc :+ valid_accuracy
     }
     
    
    val trainingacc = "training accuracy\n" + train_acc
    val testacc = "validation accuracy\n" + valid_acc
    
    //get values for plotting ROC curve
    val rocmetrics = new BinaryClassificationMetrics(predictions
      .select(col("probability"),col("label"))
      .rdd.map(r=>(r.getAs[DenseVector](0)(1),r.getDouble(1))))
    
    rocmetrics.roc().coalesce(1, false).saveAsTextFile(args(2))
    
    //save all the necessary values
    val listtoprint = List(trtime, rawacc, pre1, pre2, rec1, rec2, fpr1, fpr2, fmes1, fmes2,  auc, confmat, trainingacc, testacc, modelparam)
    val lp = sc.parallelize(listtoprint)
    lp.saveAsTextFile(args(3))**/
    
    //pipeline method - training with cross-validator
    
    val classifier = new RandomForestClassifier()
    val evaluator = new BinaryClassificationEvaluator().setLabelCol("label")
    
    //create hyperparameters grid
    val paramGrid = new ParamGridBuilder()
      .addGrid(classifier.maxBins, Array(2))
      .addGrid(classifier.maxDepth, Array(5))
      .addGrid(classifier.numTrees, Array(8))
      .addGrid(classifier.minInstancesPerNode, Array(1))
      .addGrid(classifier.featureSubsetStrategy, Array("auto"))
      .addGrid(classifier.impurity, Array("entropy"))
      .build()
      
    //define pipeline stages
    val steps: Array[PipelineStage] = Array(classifier)
    val pipeline = new Pipeline().setStages(steps)
    
    //define a cross validator
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)
      .setParallelism(1)
      
    //time starts
    val startTime = System.nanoTime() 
    
    //run the training for all combinations of those parameters
    //val pipelineFittedModel = cv.fit(trainingData)
    val pipelineFittedModel = cv.fit(dd)
    
    //training time
    val elapsedTime = (System.nanoTime() - startTime) / 1e9
    val trtime = "Training time: " +  elapsedTime + " seconds"
    //println(s"Training time: $elapsedTime seconds")

    //use the best performing model on test data
    val predictions = pipelineFittedModel.transform(testData)
    
    //get the best model
    val model = pipelineFittedModel.bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel].stages.head.asInstanceOf[RandomForestClassificationModel]
    
    //get accuracy
    val accuracy = evaluator.evaluate(predictions)
    val rawacc = "Accuracy\n" + accuracy
    //println("accuracy after pipeline fitting" + accuracy)
    
    //get feature ranking for pipeline method and save it--
    
    val importances = model.featureImportances
    val res = featureCols.zip(importances.toArray).sortBy(-_._2)
    val imf = sc.parallelize(res)
    imf.saveAsTextFile(args(1))
    
    //println(pipelineFittedModel.bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel].stages.head.asInstanceOf[RandomForestClassificationModel].extractParamMap())
    
    //see the hyperparameters key-value
    val modelparam = "model parameters:\n" + model.extractParamMap()
    
    //print performance metrics
    val labelAndPreds = predictions.select("prediction", "label").rdd.map(x =>
        (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))
    
    val metrics = new BinaryClassificationMetrics(labelAndPreds)
    val cm = new MulticlassMetrics(labelAndPreds)
    
    val auc = "AUC\n" + metrics.areaUnderROC()
    val confmat = "Confusion Matrix\n" + cm.confusionMatrix
    
    //println(metrics.areaUnderROC())
    //println(cm.confusionMatrix)
    
    
    // Precision by label
    val mlabels = cm.labels
    val f = mlabels.array
    
    val pre1 = "Precision(" + f(0).toInt + ")" + " = " + cm.precision(f(0))
    val pre2 = "Precision(" + f(1).toInt + ")" + " = " + cm.precision(f(1))
    
    //Recall by label
    val rec1 = "Recall(" + f(0).toInt + ")" + " = " + cm.recall(f(0))
    val rec2 = "Recall(" + f(1).toInt + ")" + " = " + cm.recall(f(1))
    
    //False positive rate by label
    val fpr1 = "FPR(" + f(0).toInt + ")" + " = " + cm.falsePositiveRate(f(0))
    val fpr2 = "FPR(" + f(1).toInt + ")" + " = " + cm.falsePositiveRate(f(1))
    
    //F-measure by label
    val fmes1 = "F1-Score(" + f(0).toInt + ")" + " = " + cm.fMeasure(f(0))
    val fmes2 = "F1-Score(" + f(1).toInt + ")" + " = " + cm.fMeasure(f(1))
     
    //get pointer to the model 
    val rfModel = model
   
     
    //get values for plotting ROC curve
    val rocmetrics = new BinaryClassificationMetrics(predictions
      .select(col("probability"),col("label"))
      .rdd.map(r=>(r.getAs[DenseVector](0)(1),r.getDouble(1))))
    
    rocmetrics.roc().coalesce(1, false).saveAsTextFile(args(2))
    
    //save all the necessary values
    val listtoprint = List(trtime, rawacc, pre1, pre2, rec1, rec2, fpr1, fpr2, fmes1, fmes2,  auc, confmat, modelparam)
    val lp = sc.parallelize(listtoprint)
    lp.saveAsTextFile(args(3))
  }
}

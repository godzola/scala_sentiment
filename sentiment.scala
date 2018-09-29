// Databricks notebook source
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{StopWordsRemover, Tokenizer, HashingTF, IDF}
import org.apache.spark.ml.tuning.{CrossValidator,ParamGridBuilder}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.linalg.Vector

import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

// COMMAND ----------

//
//  Sorry - can't figure out how to generate a markdown cell
//
//  Main takeaway from the next cell is you'll have to change the file path to the location of the data.
//  It will obviously depend on where you put it on your spark cluster

// COMMAND ----------

val raw_data_df = spark.read.format("csv").option("header", "true").load("/FileStore/tables/data.csv")
raw_data_df.printSchema
raw_data_df.show(5)

// COMMAND ----------

// We want to turn the label into a double, not a string AND - spark ml is wonky, it needs to be called "label"

val raw_data_mod_df = raw_data_df.withColumn("label", $"sentiment".cast(DoubleType))
raw_data_mod_df.printSchema
raw_data_mod_df.show(5)

// COMMAND ----------

// Let's look at the dataa little before we start

val pos_sent = raw_data_mod_df.filter($"label" === 1.0)
val neg_sent = raw_data_mod_df.filter($"label" === 0.0)

println("pos observations = " + pos_sent.count)
println("neg observations = " + neg_sent.count)
println("total observations = " + raw_data_mod_df.count)

neg_sent.show(10, false)


// COMMAND ----------

// Comment - ouch we have a lot of negative examples here. Before we start cutting down the sample size, or otherwise fixing the data
//
//  We have 90% of the data representing the negative class! So if we made a random number based classifier that chose numbers between 1 and 10
//  and said "positive" if we generate a 1, given random input we might actually do OK... however 
//
//  It would be funny to think that the typical state of the workd is that people generally don't like things, which may be true, or
//  that maybe we just have bad data. In either case we need to think about the potential for correction. If the state of the world is "negative"
//  we will still want to correctly find the positive examples, and if we have collected bad data, or task is to still work woth it.
//
//  So typically, you would think of weighting the positive examples by some factor to increase their impact, or maybe limiting the number of negatove samples
//  in the data. My gut feeling is, though, to first run our data and try and understand what we're working with
//
//  To that end - 4 other things come to mind 
//
// 1.  I think that - for evaluation - we want to look at something like a confusion matrix and see what kind of split we get in terms of false positives
//  So for Spark ML we can *easily* do a ROC/AUC analysis which will plot the true positives against the false positives
//
// 2.  We probably want to look at the number of positive cases we get in our train/test sets when we do our split
//
// 3.  We'll want to do some cross validation and see if we can improve the model on whatever split we initially come up with
//
// 4.  We should change the size of the splits, like 50-50 or 80-20 instead of 90-10
//

// COMMAND ----------


// Next, we'll make our pipeline

val tokenizer = new Tokenizer().setInputCol("content").setOutputCol("tokenized")
val sw_remover = new StopWordsRemover().setInputCol("tokenized").setOutputCol("sw_filtered").setCaseSensitive(false)
val tf = new HashingTF().setNumFeatures(1000).setInputCol("sw_filtered").setOutputCol("text_features")
val idf = new IDF().setInputCol("text_features").setOutputCol("features").setMinDocFreq(0)
val lr = new LogisticRegression().setRegParam(0.01).setThreshold(0.5)
val pipeline = new Pipeline().setStages(Array(tokenizer, sw_remover, tf, idf, lr))


// COMMAND ----------

// Break up the data into training and test data, a 90/10 split

val Array(train_data, test_data) = raw_data_mod_df.randomSplit(Array(0.9, 0.1), seed = 12345)



// COMMAND ----------

//  This is to see how lucky we got in our initial split
//

val pos_train = train_data.filter($"label" === 1.0)
val neg_train = train_data.filter($"label" === 0.0)
val pos_test = test_data.filter($"label" === 1.0)
val neg_test = test_data.filter($"label" === 0.0)

println("pos samples in training data: " + pos_train.count)
println("neg samples in training data: " + neg_train.count)
println("pos samples in test data: " + pos_test.count)
println("neg samples in test data: " + neg_test.count)


// COMMAND ----------

// Well, the numbers look reasonable, actually, so now let's fit the model and make our predictions

val model = pipeline.fit(train_data)
val predictions = model.transform(test_data)

// COMMAND ----------

// Let's see how we did

val eval = new BinaryClassificationEvaluator().setMetricName("areaUnderROC")
println("AUC = " + eval.evaluate(predictions))

// COMMAND ----------

// Let's do cross validation and see if we can improve on our AUC
//
// We'll work with our features, add the number we use and raise the minimum document frequency.
// I should note, the reason we started with a small-ish number of features is to avoid over-fitting

val hyper_param_grid = new ParamGridBuilder()
                           .addGrid(tf.numFeatures, Array(1000, 10000, 100000))
                           .addGrid(idf.minDocFreq, Array(0, 10, 50))
                           .build()

// COMMAND ----------

val xvalidator = new CrossValidator().setEstimator(pipeline)
                                     .setEvaluator(eval)
                                     .setEstimatorParamMaps(hyper_param_grid)
                                     .setNumFolds(2)
val xval_model = xvalidator.fit(train_data)
val xval_predictions = xval_model.transform(test_data)
println("Best XVal AUC = " + eval.evaluate(xval_predictions))

// COMMAND ----------

// OK the cross validation took 4 minutes to run, but we get a mych better performance:
//
//    Best XVal AUC = 0.9606357412264501
//
//  thats 96 percent or a 9% increase over the first try. I'd say that's pretty good.
//
//  I am curious if an RNN would do any better, the literature doesn't suggest it would, but it may.
//  Spark ML - I don't think - has a deep learning neural net. I think they use dl4j, which is a nice 
//  library. 


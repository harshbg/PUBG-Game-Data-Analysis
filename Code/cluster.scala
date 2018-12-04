import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.classification.BinaryLogisticRegressionSummary
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator



val inputfile = sql("select * from pubg_new")
val Data=inputfile.select(inputfile("winorlose").as("label"),$"boosts",$"damageDealt",$"DBNOs",$"headshotKills",$"heals",$"killPlace",$"killPoints",$"kills",$"killStreaks",$"longestKill",$"maxPlace",$"numGroups",$"revives",$"rideDistance",$"roadKills",$"swimDistance",$"teamKills",$"vehicleDestroys",$"walkDistance",$"weaponsacquired",$"winpoints",$"winorlose",$"winquartiles")

val assembler = new VectorAssembler().setInputCols(Array("boosts","damageDealt","DBNOs","headshotKills","heals","killPlace","killPoints","kills","killStreaks","longestKill","maxPlace","numGroups","revives","rideDistance","roadKills","swimDistance","teamKills","vehicleDestroys","walkDistance","weaponsacquired","winpoints","winorlose","winquartiles")).setOutputCol("features")
val data1 = assembler.transform(Data).select($"label",$"features")
val kmeans = new KMeans().setPredictionCol("cluster").setFeaturesCol("features").setK(5).setInitSteps(40).setMaxIter(99) 
val kmodel = kmeans.fit(data1)
println(s"3,${kmodel.computeCost(data1)}") 
println("Cluster centroids:")
kmodel.clusterCenters.foreach(println)
println(s"$3,${kmodel.computeCost(data1)}")
val predictions = kmodel.summary.predictions
predictions.orderBy("cluster").show()
predictions.count()

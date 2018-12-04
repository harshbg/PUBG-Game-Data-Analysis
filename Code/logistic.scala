
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.classification.BinaryLogisticRegressionSummary




val inputfile = sql("select * from pubg_new")
val Data=inputfile.select(inputfile("winorlose").as("label"),$"boosts",$"damageDealt",$"DBNOs",$"headshotKills",$"heals",$"killPlace",$"killPoints",$"kills",$"killStreaks",$"longestKill",$"maxPlace",$"numGroups",$"revives",$"rideDistance",$"roadKills",$"swimDistance",$"teamKills",$"vehicleDestroys",$"walkDistance",$"weaponsacquired",$"winpoints",$"winorlose",$"winquartiles")
val assembler = new VectorAssembler().setInputCols(Array("boosts","damageDealt","DBNOs","headshotKills","heals","killPlace","killPoints","kills","killStreaks","longestKill","maxPlace","numGroups","revives","rideDistance","roadKills","swimDistance","teamKills","vehicleDestroys","walkDistance","weaponsacquired","winpoints","winorlose","winquartiles")).setOutputCol("features")
val data1 = assembler.transform(Data).select($"label",$"features")
val Array(training, test) = data1.randomSplit(Array(0.6, 0.4), seed = 11L)
training.cache()
val lg = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
val lgModel = lg.fit(data1)
println(s"Coefficients: ${lgModel.coefficients} Intercept: ${lgModel.intercept}")




val trainingSummary = lgModel.summary
val objectiveHistory = trainingSummary.objectiveHistory
println("objectiveHistory:")
objectiveHistory.foreach(loss => println(loss))
val binarySummary = trainingSummary.asInstanceOf[BinaryLogisticRegressionSummary]

val roc = binarySummary.roc
roc.show()
println(s"areaUnderROC: ${binarySummary.areaUnderROC}")
val fMeasure = binarySummary.fMeasureByThreshold
val maxFMeasure = fMeasure.select(max("F-Measure")).head().getDouble(0)
val bestThreshold = fMeasure.where($"F-Measure" === maxFMeasure).select("threshold").head().getDouble(0)
lgModel.setThreshold(bestThreshold)


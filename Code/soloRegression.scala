import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature
import org.apache.spark.ml.feature.StandardScaler
val inputfile = sql("select * from pubg_new where match_type1 ='solo'")
val Data=inputfile.select(inputfile("winplaceperc").as("label"),$"boosts",$"damageDealt",$"DBNOs",$"headshotKills",$"heals",$"killPlace",$"killPoints",$"kills",$"killStreaks",$"longestKill",$"maxPlace",$"revives",$"rideDistance",$"roadKills",$"swimDistance",$"teamKills",$"vehicleDestroys",$"walkDistance",$"weaponsacquired",$"winpoints",$"winorlose")
val assembler = new VectorAssembler().setInputCols(Array("boosts","damageDealt","DBNOs","headshotKills","heals","killPlace","killPoints","kills","killStreaks","longestKill","maxPlace","revives","rideDistance","roadKills","swimDistance","teamKills","vehicleDestroys","walkDistance","weaponsacquired","winpoints","winorlose")).setOutputCol("features")
val data1 = assembler.transform(Data).select($"label",$"features")
val lr = new LinearRegression()
val lrModel = lr.fit(data1)
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
val trainingSummary = lrModel.summary
println(s"numIterations: ${trainingSummary.totalIterations}")
println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
trainingSummary.residuals.show()
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"MSE: ${trainingSummary.meanSquaredError}")
println(s"r2: ${trainingSummary.r2}")

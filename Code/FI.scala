import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.regression.{ RandomForestRegressor, RandomForestRegressionModel }
import org.apache.spark.ml.feature.StandardScaler
val inputfile = sql("select * from pubg_new")
val Data=inputfile.select(inputfile("winplaceperc").as("label"),$"boosts",$"damageDealt",$"DBNOs",$"headshotKills",$"heals",$"killPlace",$"killPoints",$"kills",$"killStreaks",$"longestKill",$"maxPlace",$"numGroups",$"revives",$"rideDistance",$"roadKills",$"swimDistance",$"teamKills",$"vehicleDestroys",$"walkDistance",$"weaponsacquired",$"winpoints")
val assembler = new VectorAssembler().setInputCols(Array("boosts","damageDealt","DBNOs","headshotKills","heals","killPlace","killPoints","kills","killStreaks","longestKill","maxPlace","numGroups","revives","rideDistance","roadKills","swimDistance","teamKills","vehicleDestroys","walkDistance","weaponsacquired","winpoints")).setOutputCol("features")
val data1 = assembler.transform(Data).select($"label",$"features")



val rf = new RandomForestRegressor
val model: RandomForestRegressionModel = rf.fit(data1)
// GET FEATURE IMPORTANCE
val featImp = model.featureImportances
val featureMetadata = data1.schema("features").metadata

package oata;

import scala.Tuple2;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.SparkConf;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class JZCollaborativeFilter {
    public static void main(String[] args) {
	//test1();
	test2();
    }
    public static void test1() {
        /*System.out.println("Hello World");
	String logFile = "../data/test/1.txt"; // Should be some file on your system
	SparkConf conf = new SparkConf().setAppName("Simple Application");
	JavaSparkContext sc = new JavaSparkContext(conf);
	JavaRDD<String> logData = sc.textFile(logFile).cache();

	long numAs = logData.filter(new Function<String, Boolean>() {
		public Boolean call(String s) { return s.contains("a"); }
	    }).count();

	long numBs = logData.filter(new Function<String, Boolean>() {
		public Boolean call(String s) { return s.contains("b"); }
	    }).count();

	    System.out.println("Lines with a: " + numAs + ", lines with b: " + numBs);*/
    }
    public static void test2() {
	SparkConf conf = new SparkConf().setAppName("Java Collaborative Filtering Example0");

	// Load and parse the data
	String path = "../data/test/user_log_format1_0.csv";
	SparkSession session = SparkSession.builder().config(conf).getOrCreate();
	Dataset<Row> df = session.read()
                .format("com.databricks.spark.csv")
                .option("inferSchema", "true")
                .option("header", "true")
                .load(path);
	//Dataset<Row> df = session.read().csv(path);
        df.printSchema();
	System.out.println("Rows = " + df.count());
	
	
    }
    public static void test3() {
	/*SparkConf conf = new SparkConf().setAppName("Java Collaborative Filtering Example");
	JavaSparkContext jsc = new JavaSparkContext(conf);

	// Load and parse the data
	String path = "data/mllib/als/test.data";
	JavaRDD<String> data = jsc.textFile(path);
	JavaRDD<Rating> ratings = data.map(
					   new Function<String, Rating>() {
					       public Rating call(String s) {
						   String[] sarray = s.split(",");
						   return new Rating(Integer.parseInt(sarray[0]), Integer.parseInt(sarray[1]),
								     Double.parseDouble(sarray[2]));
					       }
					   }
					   );

	// Build the recommendation model using ALS
	int rank = 10;
	int numIterations = 10;
	MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(ratings), rank, numIterations, 0.01);

	// Evaluate the model on rating data
	JavaRDD<Tuple2<Object, Object>> userProducts = ratings.map(
								   new Function<Rating, Tuple2<Object, Object>>() {
								       public Tuple2<Object, Object> call(Rating r) {
									   return new Tuple2<Object, Object>(r.user(), r.product());
								       }
								   }
								   );
	JavaPairRDD<Tuple2<Integer, Integer>, Double> predictions =
	    JavaPairRDD.fromJavaRDD(
				    model.predict(JavaRDD.toRDD(userProducts)).toJavaRDD().map(
											       new Function<Rating, Tuple2<Tuple2<Integer, Integer>, Double>>() {
												   public Tuple2<Tuple2<Integer, Integer>, Double> call(Rating r){
												       return new Tuple2<>(new Tuple2<>(r.user(), r.product()), r.rating());
												   }
											       }
																		       ));
	JavaRDD<Tuple2<Double, Double>> ratesAndPreds =
	    JavaPairRDD.fromJavaRDD(ratings.map(
						new Function<Rating, Tuple2<Tuple2<Integer, Integer>, Double>>() {
						    public Tuple2<Tuple2<Integer, Integer>, Double> call(Rating r){
							return new Tuple2<>(new Tuple2<>(r.user(), r.product()), r.rating());
						    }
						}
						)).join(predictions).values();
	double MSE = JavaDoubleRDD.fromRDD(ratesAndPreds.map(
							     new Function<Tuple2<Double, Double>, Object>() {
								 public Object call(Tuple2<Double, Double> pair) {
								     Double err = pair._1() - pair._2();
								     return err * err;
								 }
							     }
							     ).rdd()).mean();
	System.out.println("Mean Squared Error = " + MSE);

	// Save and load model
	model.save(jsc.sc(), "target/tmp/myCollaborativeFilter");
	MatrixFactorizationModel sameModel =
	MatrixFactorizationModel.load(jsc.sc(),"target/tmp/myCollaborativeFilter");*/
    }
}

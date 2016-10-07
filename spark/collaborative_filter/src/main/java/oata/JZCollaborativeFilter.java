package oata;

import scala.Tuple2;

import java.io.Serializable;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.functions;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;

public class JZCollaborativeFilter {
    public static class MovieData implements Serializable {
	private String userId;
	private String itemId;
	private String rating;
	private String timestamp;
	
	public String getUserId() {
	    return userId;
	}

	public void setUserId(String userId) {
	    this.userId = userId;
	}

	public String getItemId() {
	    return itemId;
	}

	public void setItemId(String itemId) {
	    this.itemId = itemId;
	}

	public String getRating() {
	    return rating;
	}

	public void setRating(String rating) {
	    this.rating = rating;
	}

	public String getTimestamp() {
	    return timestamp;
	}

	public void setTimestamp(String timestamp) {
	    this.timestamp = timestamp;
	}
    }

    public static class MovieData2 implements Serializable {
	private String userId;
	private String itemId;
	private double rating;
	private double timestamp;
	
	public String getUserId() {
	    return userId;
	}

	public void setUserId(String userId) {
	    this.userId = userId;
	}

	public String getItemId() {
	    return itemId;
	}

	public void setItemId(String itemId) {
	    this.itemId = itemId;
	}

	public double getRating() {
	    return rating;
	}

	public void setRating(double rating) {
	    this.rating = rating;
	}

	public double getTimestamp() {
	    return timestamp;
	}

	public void setTimestamp(double timestamp) {
	    this.timestamp = timestamp;
	}
    }
    
    public static void main(String[] args) {
	//test1();
	//test2();
	//test3();
	test4();
    }
    public static void closeInfoLog() {
	Logger.getRootLogger().setLevel(Level.WARN);
    }
    public static void test1() {
	SparkConf conf = new SparkConf().setAppName("Java Collaborative Filtering Example0");
	SparkContext sc = SparkContext.getOrCreate(conf);
	sc.setLogLevel("WARN");

	// Load and parse the data
	String path = "../data/test/ml-100k/u1.base";
	SparkSession session = SparkSession.builder().config(conf).getOrCreate();
	Dataset<Row> df = session.read().text(path);
	
	System.out.println("Super: Rows = " + df.count());
	System.out.println("Super: Row 1 = " + df.first().toString());
	System.out.println("X Man: Row 1 = " + df.first().getString(0));
	Dataset<String> df2 = df.map(new MapFunction<Row, String>() {
					public String call(Row row) {
					    String[] sarray = row.getString(0).split("\t");
					    return sarray[0];
					}
	    },Encoders.STRING());
	System.out.println("Super: Row 1 :: 0 = " + df2.first().toString());
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
		SparkConf conf = new SparkConf().setAppName("Java Collaborative Filtering Example0");
	SparkContext sc = SparkContext.getOrCreate(conf);
	sc.setLogLevel("WARN");

	// Load and parse the data
	String path = "../data/test/ml-100k/u1.base";
	SparkSession session = SparkSession.builder().config(conf).getOrCreate();
	Dataset<Row> df = session.read().text(path);
	Encoder<MovieData> mdEncoder = Encoders.bean(MovieData.class);
	//Dataset<MovieData> df2 = df.as(mdEncoder);
	Dataset<MovieData> df2 = df.map(new MapFunction<Row, MovieData>() {
					public MovieData call(Row row) {
					    MovieData md = new MovieData();
					    String[] sarray = row.getString(0).split("\t");
					    md.userId = sarray[0];
					    md.itemId = sarray[1];
					    md.rating = sarray[2];
					    md.timestamp = sarray[3];
					    return md;
					}
	    }, mdEncoder);
	MovieData md = df2.first();
	System.out.println("DF2: Row1 = " + md.userId + "::" + md.itemId + "::" + md.rating + "::" + md.timestamp);
	df2.createOrReplaceTempView("Movie");
	Dataset<Row> df3 = session.sql("SELECT userId FROM Movie group by userId");
	System.out.println("DF3: Rows = " + df3.count());
    }
    public static void test4() {
	SparkConf conf = new SparkConf().setAppName("Java Collaborative Filtering Example0");
	SparkContext sc = SparkContext.getOrCreate(conf);
	sc.setLogLevel("WARN");

	// Load and parse the data
	String path = "../data/test/ml-100k/u1.base";
	SparkSession session = SparkSession.builder().config(conf).getOrCreate();
	Dataset<Row> df = session.read().text(path);
	Encoder<MovieData2> mdEncoder = Encoders.bean(MovieData2.class);
	//Dataset<MovieData> df2 = df.as(mdEncoder);
	Dataset<MovieData2> df2 = df.map(new MapFunction<Row, MovieData2>() {
					public MovieData2 call(Row row) {
					    MovieData2 md = new MovieData2();
					    String[] sarray = row.getString(0).split("\t");
					    md.userId = sarray[0];
					    md.itemId = sarray[1];
					    md.rating = Double.parseDouble(sarray[2]);
					    md.timestamp = Double.parseDouble(sarray[3]);
					    return md;
					}
	    }, mdEncoder);
	MovieData2 md = df2.first();
	System.out.println("DF2: Row1 = " + md.userId + "::" + md.itemId + "::" + md.rating + "::" + md.timestamp);
	df2.createOrReplaceTempView("Movie");
	Dataset<Row> df3 = session.sql("SELECT userId FROM Movie group by userId");
	System.out.println("DF3: Rows = " + df3.count());
	df2.describe("rating").show();
	Dataset<Row> df4 = df2.agg(functions.countDistinct("userId"));
	df4.show();
	
    }
    public static void test5() {
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

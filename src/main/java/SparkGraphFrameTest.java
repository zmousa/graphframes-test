import model.Relation;
import model.User;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.graphframes.GraphFrame;

import java.util.ArrayList;
import java.util.List;

public class SparkGraphFrameTest {

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("SparkGraphFrame")
                .master("local[*]")
                .getOrCreate();
        spark.sparkContext().setCheckpointDir("/tmp");
        spark.sparkContext().setLogLevel("ERROR");

        List<User> uList = new ArrayList<>();
        uList.add(new User("101", "Trina", 27));
        uList.add(new User("201", "Raman", 45));
        uList.add(new User("301", "Ajay", 32));
        uList.add(new User("401", "Sima", 23));

        // Creating vertex DataFrame
        Dataset<Row> verDF = spark.createDataFrame(uList, User.class);

        List<Relation> rList = new ArrayList<>();
        rList.add(new Relation("101", "301", "Colleague"));
        rList.add(new Relation("101", "401", "Friends"));
        rList.add(new Relation("401", "201", "Reports"));
        rList.add(new Relation("301", "201", "Reports"));
        rList.add(new Relation("201", "101", "Reports"));

        // Creating edge DataFrame
        Dataset<Row> edgDF = spark.createDataFrame(rList, Relation.class);

        // Creating a GraphFrame
        GraphFrame g = new GraphFrame(verDF,edgDF);

        // Basic GraphFrame and DataFrame queries
        g.vertices().show();
        g.edges().show();

        // Get a DataFrame with columns "id" and "inDeg" (in-degree)
        g.inDegrees().show();

        // Find the youngest user’s age in the graph by querying vertex DataFrame
        g.vertices().groupBy().min("age").show();

        // Find the number of “Friends” relationship in the graph by querying edge DataFrame
        long numFriends = g.edges().filter("relationship = 'Friends'").count();
        System.out.println("Print total count of Friends relationship :: "+ numFriends);

        // Motif finding refers to searching for structural patterns in a graph
        Dataset<Row> motifs = g.find("(a)-[e]->(b)");
        motifs.filter("b.age>40").show();

        // Sub-graphs creation based on vertex and edge filters
        Dataset<Row> v2 = g.vertices().filter("age > 30");
        Dataset<Row> e2 = g.edges().filter("relationship = 'Reports'");
        GraphFrame g2 = new GraphFrame(v2,e2);
        g2.vertices().show();
        g2.edges().show();

        // Complex subgraph : triplet filters
        Dataset<Row> paths = g.find("(a)-[e]->(b)")
                .filter("e.relationship = 'Reports'")
                .filter("a.age < b.age");
        Dataset<Row> e3 = paths.select("e.src", "e.dst", "e.relationship");
        GraphFrame g3 = new GraphFrame(g.vertices(),e3);
        g3.vertices().show();
        g3.edges().show();

        // Breadth-first search (BFS)
        Dataset<Row> paths1 = g.bfs().fromExpr("name = 'Trina'").toExpr("age > 27").run();
        paths1.show();

        Dataset<Row> paths2 = g.bfs().fromExpr("name = 'Trina'").toExpr("age > 30")
                .edgeFilter("relationship != 'Colleague'")
                .maxPathLength(3)
                .run();
        paths2.show();

        // Connected components
        Dataset<Row> result = g.connectedComponents().run();
        result.select("id", "component").orderBy("component").show();

        // Label Propagation Algorithm (LPA)
        Dataset<Row> result1 = g.labelPropagation().maxIter(5).run();
        result1.select("id", "label").show();

        // PageRank
        GraphFrame results = g.pageRank().resetProbability(0.15).tol(0.01).run();
        // Display resulting pageranks and final edge weights
        results.vertices().select("id", "pagerank").show();
        results.edges().select("src", "dst", "weight").show();

        // Shortest Paths
        ArrayList<Object> l1st = new ArrayList<Object>();
        l1st.add("101");
        l1st.add("401");
        Dataset<Row> results1 = g.shortestPaths().landmarks(l1st).run();
        results1.select("id", "distances").show();

        // Triangle count
        Dataset<Row> results2 = g.triangleCount().run();
        results2.select("id", "count").show();

        /*
        // Saving and loading GraphFrames
        // Save vertices and edges as Parquet to some location.
        g.vertices().write().parquet("hdfs://myLocation/vertices");
        g.edges().write().parquet("hdfs://myLocation/edges");
        // Load the vertices and edges back.
        Dataset<Row> sameV = sqlContext.read().parquet("hdfs://myLocation/vertices");
        Dataset<Row> sameE = sqlContext.read().parquet("hdfs://myLocation/edges");
        // Create an identical GraphFrame.
        GraphFrame sameG =new GraphFrame(sameV, sameE);
        */
    }
}

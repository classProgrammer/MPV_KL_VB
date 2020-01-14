package klausur.sum

import scala.concurrent.{Await, Future}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration.Duration

object Functions {
  // given
  // def sum(seq: Seq[Int]) : Future[Int]
  // def partSum(seq: Seq[Int]) : ???

  def sum(seq: Seq[Int]) : Future[Int] = Future { seq.sum }

  def partSum(seq: Seq[Int]): Future[Int] = {
    val parts = seq splitAt(seq.length / 2)
    val right: Future[Int] = sum(parts._1)
    val left: Future[Int] = sum(parts._2)

    // t._1.flatMap(r => t._2.map(l => r + l))
    for(r <- right; l <- left) yield r + l
  }

}

object Main extends App {
  import Functions._

  val f = partSum(Seq(10,10,10,10,10))
  f.foreach(v => println(v))
  f.failed.foreach(ex => println(ex))

  Await.ready(f, Duration.Inf)
  Thread.sleep(50)
}

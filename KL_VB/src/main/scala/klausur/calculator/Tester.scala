package klausur.calculator

import akka.actor.{Actor, ActorSystem, Props}

class Adder(val seq: Seq[Int]) extends Actor{
  context.parent ! seq.sum
  override def receive: Receive = {
    case _ =>
  }
}

class Calculator(seq: Seq[Int]) extends Actor{
  var sum = 0;
  val parts = seq.splitAt(seq.length / 2)

  val adder1 = context.actorOf(Props(classOf[Adder], parts._1),"adder1")
  val adder2 = context.actorOf(Props(classOf[Adder], parts._2),"adder2")

  override def receive: Receive = {
    case v: Int => sum += v; context.become(finalBehavior)
  }

  def finalBehavior : Receive = {
    case v: Int => sum += v; context.parent ! sum; context.stop(self)
    case _ => println("Wos wüst?")
  }

}
class TestActorParent extends Actor {
  val adder1 = context.actorOf(Props(classOf[Calculator], 0 to 100), "calc")
  override def receive = {
    case x:Int => println(x)
    case _ => println("Wü i net, hob i net, brauch i net, geh hoam Kasperl!")
  }
}

object Tester extends App {
  val s = ActorSystem("MyTestSystem")
  val c = s.actorOf(Props(classOf[TestActorParent]),"TestActorParent")
  Thread.sleep(1000)
  s.terminate()
}
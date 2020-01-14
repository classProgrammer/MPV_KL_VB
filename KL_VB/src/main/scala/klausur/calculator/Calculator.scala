package klausur.calculator

import akka.actor.{Actor, Props}

class Calculator(seq: Seq[Int]) extends Actor{
  println("Calculator")
  val adder = context.actorOf(Props(new Adder(seq)), "AdderActorChild")

  override def receive: Receive = {
    case value: Int =>
      println(s"Addition result = $value")
      context.parent ! value
      println(s"Addition result sent to parent")
      context.stop(self)
  }

  override def unhandled(message: Any): Unit = {
    println(s"   === UNHANDLED: ${self.path.name}: '$message'")
  }
}

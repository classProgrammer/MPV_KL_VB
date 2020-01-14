package klausur.calculator

import akka.actor.{Actor, Props, Terminated}

import scala.concurrent.Await
import scala.concurrent.duration.Duration

class Parent extends Actor {
  val calc = context.actorOf(Props(new Calculator(1 to 100)), "Calculatorio")

  context watch calc

  override def receive: Receive = {
    case v: Int => println(s"sum = $v")
    case Terminated(actor) =>
      println("child terminated")
      context.system.terminate()
  }
}

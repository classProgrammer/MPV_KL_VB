package klausur.calculator

import akka.actor.Actor

class Adder(val seq: Seq[Int]) extends Actor{
  context.parent ! seq.sum

  override def receive: Receive = {
    case _ =>
  }
}
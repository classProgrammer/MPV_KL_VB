package klausur.calculator

import akka.actor.{ActorSystem, Props}

object Main extends App {
  val system = ActorSystem("Main")
  system.actorOf(Props[Parent], "ParentActor")

}

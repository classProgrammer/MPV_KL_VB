# Futures/Promises and Reactive Streams:

## Reactive System Def
- React to Events: Message Driven
- React to Load: Elasticity: Scale-up, scale-out
- React to Failure: Resilence: Components can fail and and recover without compromising the whole system
- React to Users: Responsiveness: System needs to react in real time under load and in the presence of failures

## Problems of Async
- Poor Thread Utilization
- Deadlocks
- Starvation of thread pools

## Fork-Join Pool
- Enables Work Stealing
- Used for ExecutionContext in scala

## Problems of Callbacks
- combining results
- nested callbacks are cumbersome
- error handling becomes confusing

## Map VS FlatMap
- Map(Type) => Future \[Type\]
- FlatMap(Future \[Type\]) => Future \[Type\]
- THEREFORE: Map(Future \[Type\]) => Future \[Future \[Type\]\] which is useless

## Of Futures and Promises
- Future = receiver of value 
- Promise: I promise there will be a value some day, sender
- Value: Option \[Try \[T\]\], where is Option or Try used in scala

## Blocking Futures
- greedy threads make shared recources unavailable for a long period of time => Thread Starvation

## Parallelism
- Control => Task Paralelism, distributed tasks
- Data => Data Paralelism, same operation on different data elements

## Reactive Streams
- Event-Driven
- Sequence of Streams
- Standard
- sync & async possible
- Design Principles
  - Handling of backpressure: Fast producers should not lead to inefficient resource
consumption.
  - Asynchronous operations: Resources should be used in parallel.


## Stream Processing
- Producer/Publisher
- Consumer/Sunscriber

## Backpressure
-  Often publishers produce items faster than subscribers can consume
-  Subscibers buffer elements
   -  high memory usage
   -  eventual program crash
- Solution:
  - Subscriber limits the number of elements the publisher is allowed to send
- Strategies
  - Drop elements
  - stop generating elements till I am ready
  - buffer elements
  - tear down stream => last option to take
  - forward backpressure to publisher

## Publisher
- process sequence of elements = like collection
- async = like Futures but with many items instead of one

## AKKA Streams VS Actors
- Streams are
  - Typesafe
  - handle backpressure automatically
  - cannot be distributed

## Pipeline
- Runnable Graph
  - Source --> Flow --> Flow --> Sink
- Source = Generate Elements
- Flow = Filter/Processing Stage
- Sink = Like Flow without output = end of the line

## Buzzword Bingo
- Runnable Graph:
  - A graph with no open input and output ports is called a runnable graph
- Partial Graph:
  - open in or output ports
- Operator Fusion: 
  - By default all processing stages are executed on the same actor
- Materialization:
  - is the process of allocating all resources to be able to run the
computation described by the graph.
- fan-in operations: 
  - multiple inputs
- fan-out operations: 
  - multiple outputs

## Error Handling
When an exception is thrown in a stage the entire stream is shut down.
- Solution:
  - Recover: Emit a final element and then complete.
  - Recover with retries: Replace stream by a new one.
  - Repeatedly restart stream.
  - Use actor supervision.

# ACTORS Slides:

## Definition
- is an object with an identity,
- that has a behavior,
- only interacts using asynchronous
message passing.
- Like a Human Being
  - They also communicate by transmitting messages
  (speaking, sending mails, etc.).
  - Transmitting message takes time.
  - Can perform activities while receiving messages.
  - Carry out activities one after the other.
  - Humans’ brains are totally “isolated”.

## Components
- Actor System
  - Hierarchical group of Actors
- Actor Class
  - Actor Template
- Actor Instance
  - Runtime Instance of Actor
- Actor Reference
  - Object used to address Actor
  - Hides information about the location of an actor
- Message
- Mailbox
  - Buffering of messages
- Dispatcher
  - Assigns compute resources to actors

## Features
- Encapsulation
  - no way to access methods of an Actor => comm. via messages
- Message Centric
  - comm only via one way messages
  - messages = immutable
- NO SYNCHRONIZATION within Actors needed
  - messages are processed sequentially
- Lightweight
  - Server can have thousands of threads but millions of actors
  
## Programming Model
- outside view = concurrent
- inside view = sequential

## Message Delivery
- Generally unreliable = messages can get lost
- Strategies:
  - at-most-once: 0..1, lost => don't care
  - at-least-once: 1..*, lost => resend, confirmation necessary
  - exactly-once: 1, lost => resend, if already processed skip, confirmation necessary

## Message Order
- guaranteed for 2 Actors
- 3 or more Actors => order will vary

## The Ask Pattern Bullshit
- Wait for result with future
- Ask Pattern uses temporary Actor

## Failure Handling in Asynchronous Systems
- Actors are Decoupled = Resilent
  - No other Actor is affected when an error occurs in an Actor
- Supervision strategy
  - Parent is the supervisor and should handle errors
    - Restart: Replace failed Actor
    - Resume: Actor shall continue wiht next message
    - Stop: Permanently stop actor
    - Escalate: Send to Supervisor
  - Akka Strategies
    - OneForOneStrategy: Action is applied only to failed child
    - AllForOneStrategy: Action is applied to all children

## Cluster
Distributed Actors can form a cluster.
Collection of nodes/actors
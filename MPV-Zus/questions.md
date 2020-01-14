## Questions Kugelschreiber
- Why is shared memory divided into banks?
  - To increase performance of vector systems
  - ![](multbanks.png)
  - Vectorsysteme können den Geschwindigkeitsnachteil durch das verwenden von mehreren memory banks wett machen
  - Geschwindigkeitsnachteil weil CPU Systeme auf der CPU auch Memory (Caches) haben und nicht immer in den Hauptspeicher schreiben müssen
- Cache and Cache Coherence
  - ![](./cache.png)
  - Vector schneller als liste durch caching
- How does the GPU hide Memory Latencies
  - ![](./workinflight.png)
- Bus Snooping
  - Cache controller snoops the bus to idnetify write operations on shared variables to keep the cache up-to-date
  - Cache
    - Fast memory close to CPU
- Block, Grid, Warp
  - Grid => contains blocks "unlimited number of blocks possible"
  - Block
    - contains Threads/Warps, has a maximum number e.g. 1024 Threads per block
    - Has an smem (shared memory)
    - All threads in a block are executed on a SM (Streaming Multiprocessor)
  - Warp => Threads are executed in groups e.g. of 32 Threads, that's called a Warp
- Moores Law
  - ![](moore.png)
  - alle 1.5-2 Jahre verdoppelt sich die Komplexität von integrierten Schaltkreisen
- SIMD
  - GPGPU
  - Single CPU exclusively for control
  - Large collection of ALUs with own small memory
  - Vector Systems
  - synchronous execution
  - 2 Sates
    - execute instruction
    - idle
  - Disadvantage
    - often idle with conditional branches
  - Pros
    - easy to program
    - scale well
    


- How to proper use __syncthreads when copying from the shared into the global memory into kernel
  - ![](synct.png)
  - ![](synct2.png)
- Chache Coherence
  - Coherent = all shared variable values in all caches are the same
    - Done by e.g. Bus Snooping
  - Incoherent = values of the shared variable vary among caches
  - Incoherence is allowed if
    - The error is detected and corrected with the next read operation

- What undefined behaviour can occur in the kernel at an improper use of __syncthreads?
  - ![](synct.png)
  - Avoid the use of __syncthreads inside divergent code.
  - Barrier jumps in divergent codes
  - Some threads might jump over the __syncthreads barrier


- Gustafson Gesetz + mathematische Beschreibung?
    - do more work in parallel/ solve bigger problem in the same time
    - speedup = 1 + (n - 1) p
    - The speedup with 1 parallel part is 1.0 which is correct since it's sequential then
    - The speedup with e.g. 70% parallel part and 2 cores is 1 + (2-1) * 0.7 = 1.7
    - 4 cores = 1 + 3 * 0.7 = 3.1
    - usw.

- Diverging Branches?
    - If-then-else, while statements => Branch
    - GPU serialisiert Threads in Warp, wenn
sich Control-Flow/Verzweigung darin befindet.

- Register vergrößern auf der GPU, welche Vorteile und Nachteile hat das?
  - Cons
    - evtl. weniger shared memory
    - weniger Threads möglich
  - Pros
    - mehr scalare speicherbar 

- Welche Speicher gibt es in einer GPU?
  - Constant => cmem
  - global => gmem
  - Local => Lmem, lmem
  - Register => reg
  - shared => smem
  - Caches = L1, L2

# Questions Heinzelmännchen
- Why is classic synchronization bad?
  - Deadlock
  - Semaphores
  - Greedy Threads block access to shared memory
  - Synchronization lot's of work
  - error prone
- Resilience, Elasticity, Reactivity and Message driven are key aspects of actors, Explain those terms?
  - Message Driven
    - React to Events
    - Publish/Subscribe
    - Communication via Messages
    - Asynchrounous
    - unreliable
      - at-most-once
      - at-least-once
      - exactly-once
  - Resilence
    - React to Failure
    - System should not be biased when a failure occurs
  - Elasticity
    - React to Load
    - System should work under varying load without problems
      - scale-up
      - scale-out
  - Responsivness
    - Ract to Users
    - System should communicate in real time with the user even when the system is under load and failures are present
    
- How do Akka Actors implement those aspects?
  - Message Driven
    - Actors handle this
      - internally
      - Via Queues
      - Message Transmission is the behaviour of the Actors
  - Resilence Handled via Policies
    - Default => Parent has to take care of the problem
    - State and behaviour are isolated
    - Ploicies
      - Escalation: Help me father, Supervisor/Parent has the problem now
      - Restart: Failed Actor is replaced by a new instance
      - Continue: Actor continues with next message
      - Stop: If Actor failes it is stopped forever



- Data Race
  - Schreiben in einen Cache wird eventuell von anderen Threads übersehen

- Explain Responsiveness and how Resilience and Elasticity contribute to it 
  - Resilence => Handeles Failures
  - Elasitcity => Handles Load
  - => Therefore both prevent the system from stopping and prevent e.g. the UI or whatever from blocking making the system responsive in any case
- Was ist der Unterschied zwischen Object- und Actor Referenz?
  - ActorRef => Referenz auf den Actor, der tatsächliche Speicherort des Actors ist verschleiert
  - Object Referenz => Referenz auf ein Objekt, Speicherort bekannt
- ForkJoinPool erklären, Vorteile, wo findet es in scala Anwendung?
  - Verwendet: ExecutionContext
    - import ...ExecutionContext.Implicits.global
  - ForkJoin: Threads können Work-Stealing betreiben
  
- Was ist das ask pattern, wie und wo wendet man es an?
  - Man kann ein Ergebnis anfragen welches in eine Future gewrappt wird. Somit can man callbacks registrieren und asynchron das Ergebins verarbeiten
  

  
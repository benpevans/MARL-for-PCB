# MARL-for-PCB

This project investigated and explored useful options for RL methodologies that are capable of solving MAPF problems; this is relevant because of the success of RL in many domains that typically are solved by heuristic search. The project aim was for the solution to be applicable to connecting multiple point pairs on PCBs.

The environment was modelled as a Dec-POMDP, where agents were only able to view a snapshot of the current state. The chosen methodology explored using the effects of sharing a Q-network and replay buffer amongst concurrently learning agents. This was executed within the framework of centralised training and decentralised execution. Experiences were shared among agents, however agents’ actions were executed locally.

This methodology is relatively computationally cheap and can scale extremely well when compared to alternative methods. The approach is similar to that described by Sartoretti et al. (2018); however, rather than imitation learning, a combination of transfer and curriculum learning was used to great success to enable agents to learn general policies.

The experiments conducted with two agents show that there is no requirement for explicit agent communication in order for agents to consistently reach their goal states. Agents learned a common, homogeneous policy whereby general strategies were shown such as the ability to avoid one another and still reach their goal state.

# 2048

AI strategies to beat [2048](https://play2048.co) (game)


## Implemented strategies:

- [x] Random
- [x] Eager search
- [x] Maximization
- [x] Minimax
- [x] Expetimax
- [x] Monte Carlo Tree Search
- [ ] ~~Markov Decision Process~~ → Infeasible (too many states)
- [ ] Deep Reinforcement Learning


## Heuristics

- Monotonicity
- Smoothness
- Free Tiles
- Inmediate merges
- Gradients
- Total sum
- Max. value

**Note:** I recommend to find the weights of each heuristic using [CMA-ES](https://en.wikipedia.org/wiki/CMA-ES)


## Interesting links


- [What is the optimal algorithm for the game 2048?
](https://stackoverflow.com/questions/22342854/what-is-the-optimal-algorithm-for-the-game-2048)
- [Efficient C++ implementation](https://github.com/nneonneo/2048-ai)
- The Mathematics of 2048 (blog series)
	- [1] [Minimum Moves to Win with Markov Chains](https://jdlm.info/articles/2017/08/05/markov-chain-2048.html)
	- [2] [Counting States with Combinatorics](https://jdlm.info/articles/2017/09/17/counting-states-combinatorics-2048.html)
	- [3] [Counting States by Exhaustive Enumeration](https://jdlm.info/articles/2017/12/10/counting-states-enumeration-2048.html)
	- [4] [Optimal Play with Markov Decision Processes](https://jdlm.info/articles/2018/03/18/markov-decision-process-2048.html)
- [Artificial Intelligence has crushed all human records in 2048. Here’s how the AI pulled it off.](http://www.randalolson.com/2015/04/27/artificial-intelligence-has-crushed-all-human-records-in-2048-heres-how-the-ai-pulled-it-off/)
- [Github benchmark](https://github.com/aszczepanski/2048)
- [Bit Twiddling Hacks
](http://graphics.stanford.edu/~seander/bithacks.html)

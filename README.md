# Poker-Internal-Search

Repo dedicated to replicating MCTS-like search in-context for the downstream task of poker.


For Internal Search, run the following:

1. Build and install the postflop solver:
   ```bash
   cd internal_search/postflop_solver_engine
   maturin develop
   cd ..
   ```

2. Create a `.env` file based on the `example.env` file:
   ```bash
   cp example.env .env
   ```

3. Edit the `.env` file and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

### Running the Game


To run a game between all CFR agent and an LLM player:

```bash
python run_two_llm_game.py
```

To run a game between two LLM players:

```bash
python run_game.py
```

### Testing the Postflop Solver

To test the postflop solver:

```python
from postflop_solver import SolverState, get_optimal_action

# Create a test state
state = SolverState(
    board_cards=["Ah", "Kh", "Qh"],  # Flop cards
    hole_cards=["Jh", "Th"],         # Player's hole cards
    pot_size=100,                     # Current pot size
    stack_sizes=[900, 900],          # Remaining stack sizes
    position=0,                       # Player's position (0 for OOP, 1 for IP)
    betting_history=["check", "bet_75"]  # History of betting actions
)

# Get the optimal action
result = get_optimal_action(state)
print(f"Optimal action: {result.action}")
print(f"Amount: {result.amount}")
print(f"Reason: {result.reason}")
```

Or use the test script:
```bash
python test_solver.py
```

### Customizing the Game

You can customize the game by changing the parameters when initializing the `MixedPlayerGame` class:

- `buyin`: The amount of chips each player starts with (default: 500)
- `big_blind`: The big blind amount (default: 5)
- `small_blind`: The small blind amount (default: 2)
- `max_players`: The maximum number of players (default: 6)
- `ai_player_ids`: The IDs of players controlled by AI (default: [0, 1])
- `openai_model`: The OpenAI model to use (default: value in .env file)
- `openai_api_key`: The OpenAI API key (default: value in .env file)
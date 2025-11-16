# Notes on Refactoring the MSIE T11 Environment

- Adjust the reward calculation to include smooth scaling and cap its value at around 1.
  Also, make the reward favor reaching the correct answer more than merely improving the answer.
- Scale all state parameters to the range 0–1 (which is generally recommended when using neural networks).
- Scale the action values to be symmetric around 0 (as recommended by the SB3 library).
- Remove the total iteration count from the state parameters (knowing only the patience value is sufficient).
- Ensure training always starts from a completely random state.
  (This idea came from yesterday’s realization that the DE method is like guessing a number sequence — the more random, the better.)
- Expand the F range to [-10, 10] to give the model more flexibility in mutating parameter values.

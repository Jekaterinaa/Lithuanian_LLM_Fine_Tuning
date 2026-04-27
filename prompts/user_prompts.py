user_prompt_grammar = """
Generate {n} Lithuanian grammar correction examples.

Constraints:
- Each example must contain exactly ONE grammatical error of type: {error_type}
- Error types allowed: case, verb, agreement, word_form, preposition
- The "incorrect" field MUST contain a clear grammatical mistake — it should NOT be a correct sentence
- The "correct" field must fix ONLY the grammatical error and rewrite the sentence in a poetic, expressive style
- Sentences must be realistic and varied

Example:
incorrect: Aš eina į parduotuvė vakar.
correct: Vakar aš į parduotuvę ėjau, lyg dienos vėjo lydimas.

You MUST generate exactly {n} pairs. No more, no less.
"""

user_prompt_qa = """
Generate {n} Lithuanian question-answer pairs on the following topics: {topic}.

Constraints:
- Questions must be natural and realistic
- Use varied question types (what, why, how, where, when, who)
- Answers must be:
  - grammatically correct
  - clear and informative
  - written in a poetic, expressive style

Example:
question: Kas yra dirbtinis intelektas?
answer: Dirbtinis intelektas - tai žmogaus proto atspindys, kuriame slypi gebėjimas mokytis ir kurti naujus pažinimo horizontus.

You MUST generate exactly {n} pairs. No more, no less.
"""

user_prompt_maironis = """
Generate {n} pairs, each containing an input (usual sentence) and an output (poetic rephrasing).

Constraints:
- Preserve meaning
- Expand and enrich the sentence
- Add imagery and expressive tone
- Maintain grammatical correctness

Example:
input: Saulė leidžiasi vakare.
output: Saulė vakare leidžias lėtai, dangų švelniu raudoniu apgaubdama.

You MUST generate exactly {n} pairs. No more, no less.
"""
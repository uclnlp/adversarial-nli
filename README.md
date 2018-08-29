# adversarial-nli

Code for generating Adversarial examples for Neural NLI Models, by violating logic constraints. All methods are described in the CoNLL 2018 paper [Adversarially Regularising Neural NLI Models to Integrate Logical Background Knowledge](https://arxiv.org/abs/1808.08609).

The idea used in this paper is the following - assume you have two sentences, *sentence one* and *sentence two*. You can ask a model whether one sentence
entails or contradicts the other, and the NLI model says "**sentence one** contradicts **sentence two**, but **sentence two** does not contradict **sentence one**".

You know the model is making a mistake, because *contradiction is symmetric*, and if **sentence one** contradicts **sentence two**, then **sentence two** should contradict **sentence one** as well - if the model does not agree on this, then it is clearly making a mistake somewhere.

Similarly, you can define other constraints/rules for other entailment relations. For instance, entailment is reflexive and transitive.

In this paper, we define:
- A continuous _inconsistency loss_ measuring to which degree a given model violates a set of constraints,
- A method for systematically generating _adversarial examples_, by violating such constraints,
- A set of _adversarial datasets_ for measuring the robustness of NLI models to such adversarial examples, and
- An adversarial training procedure for training models that are more robust to such adversarial examples.

The paper is available at https://arxiv.org/abs/1808.08609


### BibTeX

```
@inproceedings{minerviniconll18,
  author    = {Pasquale Minervini and
               Sebastian Riedel},
  title     = {Adversarially Regularising Neural NLI Models to Integrate Logical Background Knowledge},
  booktitle = {Proceedings of the 22nd Conference on Computational Natural Language Learning (CoNLL 2018)},
  publisher = {Association for Computational Linguistics},
  year      = {2018}
}
```

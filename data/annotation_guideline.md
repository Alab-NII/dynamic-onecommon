# Annotation Guideline for Spatio-temporal Expressions

## Goal

The goal of this annotation project is to detect 3 types of **spatio-temporal expressions** (previous states, movements, current state) in Dynamic-OneCommon.

## Spatio-temporal Expressions

We classify spatio-temporal expressions into **3 types** and annotate the types of expressions contained in each utterance.
Each utterance can contain multiple types of spatio-temporal expressions.
If an utterance does not contain any type, it should be annotated as **None**.
We denote the expected annotation as (Previous), (Movement), etc at the end of the utterance.

1. **Previous States**

This type makes reference to <ins>the previous positions/locations of the entities</ins>.

- *"Was"* it to the left of the previous selected dot? (Previous)

- It *"formed"* a triangle with two smaller dots *"in previous turn"*. (Previous)

- The dot *"started out"* directly below a smaller, darker dot *"before moving left"*. (Previous) (Movement)

- Did it end up near *"where the dark dot was in the previous turn"*? (Previous) (Current)

2. **Movements**

This type makes reference to <ins>how the positions of the entities (or the player's view) changed</ins>.

- The dot *"curved up to the left"*. (Movement)

- Black dot *"travels northwest (NW)"* and lands next to a tiny light dot. (Movement) (Current)

- Two dots *"moving in parallel to the right"*. (Movement)

- I see one gray dot *"following behind"* a smaller, darker dot. (Movement)

- I have a large pale grey that *"moves down"* but starts out *"curving"* to the right and then *"takes a sharp turn"* to the south east. (Movement)

3. **Current State**

This type makes reference to <ins>the current position of the entities (at the end of the current turn)</ins>.

- One dot *"ended up"* between two larger dots *"after the animation"*. (Current)

- Do you *"now"* see one large dot below and slightly to the right of *"smaller, larger"* dot? (Current)

- It *"stopped"* to the right of the three darker dots. (Current)

Positions without temporal expressions may refer to the current state *"or"* both current and previous states.
(This should be judged based on the context)

- Near a smaller black dot? (Current)

- Pick the bottom left one. (Current)

- The lower one looks slightly darker to me? (Previous) (Current)

4. **None**

If an utterance does not make reference to any specific spatio-temporal information, they should be marked as **None**.

Expressions of color, size, number, etc should be ignored:

- I see two large gray dots. (None)

- Pick the smaller one. (None)

- Yes, let's choose it. (None)

Expressions which do not contain specific spatio-temporal information should be ignored:

- Our dots will are not appear in the same locations. (None) 

- How are they located before the dots move? (None)

- Let's select the previous dot again. (None)

Expressions of **possession** should be ignored, unless they contain any specific spatio-temporal information:

- I have it. / I still see it. / I still have most of dots from previous turn. (None)

- I lost it. / The previous dot is gone. / That dot left my view but may have come back. (None)

- My dot left my view *"to the left"*. (Movement)

## Annotation Tips

Short answers/questions/acknowledgements/repairs should also be considered as spatio-temporal expressions:

1. before or after the dots move? <None>
2. after (Current)

- moving fast? (Movement)

- pick the left one (Current)

- select the bottom one again (Previous) (Current)

- sorry i meant NW instead of NE (Movement)

When you annotate, please take into account the context (e.g. dialogue history):

1. I still have the triangle from previous turn. (Previous)
2. Take the bottom left one. I still have it. (Previous)

- Dark one is on the left of the lighter dot. (Current) (Previous)?

Do not rely on superficial cues: for instance, using past tense *may not* indicate previous states.

- I still have the black from earlier. (None)

- I do too, and there was a smaller dot that passed closely on its left moving in the opposite direction. (Movement)

In ambiguous/difficult cases, use all availabe information to make the best judgements:

- The two dots came near each other and spread apart. (Movement)

- The dot moved from left to right. (Movement)

- The dot moved from bottom left to upper right in my view. (Previous) (Movement) (Current)

- The dot is moving to the right by itself. (Movement) (Previous)? (Current)?

## To do

* modify asd in other repo to have correct evidence
* check out gridded results
* compare evidence/ll of my solution, jake's solutions

--ro 19.069070174197527 --ssq 2.5019481075308638 --delta 10.526593885061727
    evidence=627341.397251
    neg. log likelihood=3119100.34894

--ro 17.238799208518522 --ssq 3.1579704282716046 --delta 23.578709766296299
    evi=764016.10162078473

## Ideas

* regress where foe moves towards eye, foe moves away from eye

## Solutions

```
--ro 7.06184735 --ssq 56.27155504 --delta 0.12

--ro 8.84497231 --ssq 9.9393613 --delta 0.51312846

--ro 8.84497231 --ssq 9.9393613 --delta 0.5
    evi=-2629.9342857768866
    nll=7444.9439632018575

--ro 8.84497231 --ssq 9.93496026  --delta 0.51151012
    evi=-2629.9265325811289
    nll=7448.1781925010782

--ro 7.06184735 --ssq 56.27155504 --delta 0.12
    evidence=-700.661732086
    neg. log likelihood=1356.99041683

--ro 9.2329678 --ssq 1.64542619  --delta 4.98630877
    TNC max dude, starting from Ridge solution
    evidence=-11298.0465625
    neg. log likelihood=44957.8554228


```

## Jake

```
Active-Set

--ro 11.0 --ssq 1.3 --delta 10.7
    evidence=2262.81097632
    neg. log likelihood=96013.442144

log these, then exp
(2.0, 20.0) -> 11
(1.0, 2*10e4) -> 1.3
(1.0, 2*10e4) -> 10.7
```

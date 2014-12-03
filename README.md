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

--ro=7.06184735 --ssq=56.27155504 --delta=0.12
    evidence=-700.661732086
    neg. log likelihood=1356.99041683

[ 9.2329678   1.64542619  4.98630877]
    TNC max dude, starting from Ridge solution

```

## Jake

```
Active-Set

--ro 11.0 --ssq 1.3 --delta 10.7

log these, then exp
(2.0, 20.0) -> 11
(1.0, 2*10e4) -> 1.3
(1.0, 2*10e4) -> 10.7
```

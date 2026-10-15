# Error analysis: qwen2.5-1.5b

## Parse-failure rate (% of outputs with no recognizable label/letter)

| condition        |   agnews |   arc |   sst2 |
|:-----------------|---------:|------:|-------:|
| all6@0           |        0 |     2 |      0 |
| all6@1           |       34 |     0 |      0 |
| capitalization@0 |        1 |     4 |      0 |
| capitalization@1 |        0 |     0 |      0 |
| cosmetic2@0      |        3 |     0 |      0 |
| cosmetic2@1      |        9 |     0 |      0 |
| cosmetic3@0      |        0 |     3 |      0 |
| cosmetic3@1      |       17 |     0 |      0 |
| mixed3@0         |        2 |     0 |      0 |
| mixed3@1         |        3 |     0 |      0 |
| original         |        0 |     3 |      0 |
| original_rerun   |        0 |     3 |      0 |
| punctuation@0    |        0 |     1 |      0 |
| punctuation@1    |        4 |     0 |      0 |
| reorder@0        |        4 |     0 |      0 |
| reorder@1        |        0 |     0 |      0 |
| semantic2@0      |        6 |     0 |      0 |
| semantic2@1      |        0 |     1 |      0 |
| semantic3@0      |        8 |     0 |      0 |
| semantic3@1      |        0 |     0 |      0 |
| synonym@0        |        3 |     2 |      0 |
| synonym@1        |        0 |     0 |      0 |
| typo@0           |        0 |     1 |      0 |
| typo@1           |        0 |     0 |      0 |
| whitespace@0     |        7 |     0 |      0 |
| whitespace@1     |        0 |     0 |      0 |

## Predicted-label distribution per condition

### sst2

| condition        |   negative |   positive |
|:-----------------|-----------:|-----------:|
| all6@0           |         60 |         40 |
| all6@1           |         71 |         29 |
| capitalization@0 |         67 |         33 |
| capitalization@1 |         62 |         38 |
| cosmetic2@0      |         65 |         35 |
| cosmetic2@1      |         63 |         37 |
| cosmetic3@0      |         63 |         37 |
| cosmetic3@1      |         62 |         38 |
| mixed3@0         |         60 |         40 |
| mixed3@1         |         62 |         38 |
| original         |         66 |         34 |
| original_rerun   |         64 |         36 |
| punctuation@0    |         63 |         37 |
| punctuation@1    |         67 |         33 |
| reorder@0        |         60 |         40 |
| reorder@1        |         62 |         38 |
| semantic2@0      |         60 |         40 |
| semantic2@1      |         62 |         38 |
| semantic3@0      |         60 |         40 |
| semantic3@1      |         62 |         38 |
| synonym@0        |         62 |         38 |
| synonym@1        |         66 |         34 |
| typo@0           |         61 |         39 |
| typo@1           |         64 |         36 |
| whitespace@0     |         64 |         36 |
| whitespace@1     |         65 |         35 |

### agnews

| condition        |   Business |   NONE |   Sci/Tech |   Sports |   World |
|:-----------------|-----------:|-------:|-----------:|---------:|--------:|
| all6@0           |         26 |      0 |         25 |        3 |      46 |
| all6@1           |         14 |     34 |          0 |       37 |      15 |
| capitalization@0 |         29 |      1 |         46 |       12 |      12 |
| capitalization@1 |         42 |      0 |         12 |       42 |       4 |
| cosmetic2@0      |         34 |      3 |         28 |       21 |      14 |
| cosmetic2@1      |         45 |      9 |          5 |       41 |       0 |
| cosmetic3@0      |         21 |      0 |         24 |        5 |      50 |
| cosmetic3@1      |         55 |     17 |          2 |       26 |       0 |
| mixed3@0         |         15 |      2 |         70 |       11 |       2 |
| mixed3@1         |          1 |      3 |         24 |       72 |       0 |
| original         |         42 |      0 |         11 |       21 |      26 |
| original_rerun   |         42 |      0 |         11 |       21 |      26 |
| punctuation@0    |         23 |      0 |          9 |        3 |      65 |
| punctuation@1    |         39 |      4 |          8 |       23 |      26 |
| reorder@0        |         42 |      4 |         16 |       23 |      15 |
| reorder@1        |         42 |      0 |         11 |       19 |      28 |
| semantic2@0      |         38 |      6 |         15 |       25 |      16 |
| semantic2@1      |         24 |      0 |         34 |       23 |      19 |
| semantic3@0      |         40 |      8 |         10 |       24 |      18 |
| semantic3@1      |         26 |      0 |         24 |       22 |      28 |
| synonym@0        |         37 |      3 |         15 |       24 |      21 |
| synonym@1        |         27 |      0 |         23 |       21 |      29 |
| typo@0           |         38 |      0 |         11 |       22 |      29 |
| typo@1           |         40 |      0 |         11 |       22 |      27 |
| whitespace@0     |         39 |      7 |         11 |       25 |      18 |
| whitespace@1     |         41 |      0 |         10 |       22 |      27 |

### arc

| condition        |   A |   B |   C |   D |   NONE |
|:-----------------|----:|----:|----:|----:|-------:|
| all6@0           |  11 |  18 |  30 |  39 |      2 |
| all6@1           |  18 |  23 |  29 |  30 |      0 |
| capitalization@0 |  19 |  26 |  27 |  24 |      4 |
| capitalization@1 |  19 |  21 |  28 |  32 |      0 |
| cosmetic2@0      |  20 |  27 |  29 |  24 |      0 |
| cosmetic2@1      |  19 |  21 |  27 |  33 |      0 |
| cosmetic3@0      |  14 |  23 |  31 |  29 |      3 |
| cosmetic3@1      |  17 |  19 |  26 |  38 |      0 |
| mixed3@0         |  19 |  24 |  32 |  25 |      0 |
| mixed3@1         |  15 |  23 |  27 |  35 |      0 |
| original         |  22 |  24 |  27 |  24 |      3 |
| original_rerun   |  21 |  25 |  26 |  25 |      3 |
| punctuation@0    |  21 |  21 |  26 |  31 |      1 |
| punctuation@1    |  21 |  28 |  29 |  22 |      0 |
| reorder@0        |  22 |  25 |  28 |  25 |      0 |
| reorder@1        |  22 |  25 |  27 |  26 |      0 |
| semantic2@0      |  22 |  24 |  28 |  26 |      0 |
| semantic2@1      |  17 |  25 |  26 |  31 |      1 |
| semantic3@0      |  21 |  23 |  29 |  27 |      0 |
| semantic3@1      |  19 |  24 |  29 |  28 |      0 |
| synonym@0        |  22 |  21 |  28 |  27 |      2 |
| synonym@1        |  17 |  24 |  24 |  35 |      0 |
| typo@0           |  22 |  25 |  25 |  27 |      1 |
| typo@1           |  23 |  22 |  26 |  29 |      0 |
| whitespace@0     |  22 |  27 |  27 |  24 |      0 |
| whitespace@1     |  22 |  28 |  27 |  23 |      0 |

## SQuAD: mean output length (words) and EM per condition

| condition        |    f1 |   em |   len_words |
|:-----------------|------:|-----:|------------:|
| all6@0           | 0.685 | 0.46 |        6.66 |
| all6@1           | 0.667 | 0.4  |        7.28 |
| capitalization@0 | 0.873 | 0.76 |        3.08 |
| capitalization@1 | 0.849 | 0.7  |        3.5  |
| cosmetic2@0      | 0.856 | 0.73 |        3.06 |
| cosmetic2@1      | 0.823 | 0.63 |        4.36 |
| cosmetic3@0      | 0.863 | 0.74 |        2.82 |
| cosmetic3@1      | 0.771 | 0.62 |        4.68 |
| mixed3@0         | 0.753 | 0.53 |        5.51 |
| mixed3@1         | 0.707 | 0.48 |        6.45 |
| original         | 0.885 | 0.77 |        3.03 |
| original_rerun   | 0.889 | 0.78 |        3.02 |
| punctuation@0    | 0.821 | 0.63 |        4.49 |
| punctuation@1    | 0.846 | 0.67 |        4.16 |
| reorder@0        | 0.861 | 0.72 |        3.59 |
| reorder@1        | 0.856 | 0.7  |        3.73 |
| semantic2@0      | 0.759 | 0.56 |        5.09 |
| semantic2@1      | 0.787 | 0.59 |        5.07 |
| semantic3@0      | 0.759 | 0.54 |        5.42 |
| semantic3@1      | 0.795 | 0.61 |        4.82 |
| synonym@0        | 0.85  | 0.69 |        3.89 |
| synonym@1        | 0.88  | 0.74 |        3.49 |
| typo@0           | 0.843 | 0.71 |        3.14 |
| typo@1           | 0.903 | 0.8  |        2.85 |
| whitespace@0     | 0.861 | 0.72 |        3.49 |
| whitespace@1     | 0.857 | 0.72 |        2.85 |

## Edit magnitude vs. effect (all perturbed conditions, all tasks)

Spearman(char edit distance, drop) = 0.438 (p=8.12e-06); Spearman(char edit distance, flip rate) = 0.455 (p=3.2e-06); n=96

## Example flips (original correct -> edited wrong)

### sst2: worst condition = `all6@1`

Edited prompt instruction (first item):

```
 You ARE an  expert Attitude ASSESSMENT   aide Determine whetheer 	The overall opinion expressed in the critique is POSITIVE Or NEGATIVE 
go THROUGH   the movvie Critique provided   BELOW attentively   RESPONSE	with 	precisely 	a 	SINGLE   term: positive or negative; 





```

- `sst2-589` gold=`positive` | original→`positive` | edited→`Negative`
- `sst2-472` gold=`positive` | original→`positive` | edited→`Negative`
- `sst2-782` gold=`positive` | original→`positive` | edited→`negative`

### agnews: worst condition = `all6@1`

Edited prompt instruction (first item):

```
 You ARE an  expert News CATEGORY   categorizer Decide whicch 	Topic the article corresponds to the candidate topics ARE World. sPORTS   BUSINESS  AND SCI/TECH. 
go THROUGH   the newws Article given   BELOW carefully   REPLY	with 	solely 	the 	SUBJECT   label and no other   text. 





```

- `ag-3132` gold=`World` | original→`World` | edited→`SPOrts`
- `ag-5193` gold=`World` | original→`World` | edited→`SPOrts`
- `ag-1548` gold=`Business` | original→`Business` | edited→`World`

### arc: worst condition = `all6@0`

Edited prompt instruction (first item):

```

examine the   multiple-choice problem spuplied below carefully
answer with solely the letter of the accurate option (a b c, or d)  you	are a 	well-informwd science   techer
pck 	the	sungle most suitable 	answer  among the choices.	



```

- `Mercury_SC_405341` gold=`A` | original→`A. moisture forming on a mirror when` | edited→`The correct answer is C. Ice cubes`
- `VASoL_2007_3_1` gold=`A` | original→`A. Fly` | edited→`D.`
- `Mercury_SC_LBS10662` gold=`C` | original→`C. conduction.` | edited→`D. radiation.`

### squad: worst condition = `all6@1`

Edited prompt instruction (first item):

```
 You ARE a  precise Reading UNDERSTANDING   helper The answer 	Is a shorrt span taken straight from the PARAGRAPH 
go THROUGH   the passaage Given below   CAREFULLY and AFTERWARDS response the	Question   concerning it   ANSWER	with 	just 	the 	RESPONSE   segment, with no any   elaboration; 





```

- `56e2042ecd28a01900c67c1e` gold=`John Myhill` | original→`John Myhill` | edited→`John Myhill provided a definition of linear bounded automata`
- `5727cd7dff5b5019007d958e` gold=`third largest` | original→`third largest` | edited→`The two US Navy bases, Blount Island Command and the nearby `
- `57296d1b1d0469140077940d` gold=`a double membrane` | original→`The fact that chloroplasts are surrounded by a double membra` | edited→`The fact that chloroplasts are surrounded by a double membra`

# ml-genderBR

This project aims to advance on previous [genderBR](https://github.com/meirelesff/genderBR) algorithm used to predict gender from Brazilian first names. In its current form, the method uses data collected by the IBGE's Brazilian Census to make predictions: if a given first name is used by, say, 99% Brazilian males, then it is probable that a person with this given name is a male person. Using this simple logic, [gendeBR](https://github.com/meirelesff/genderBR) attributes gender when most uses of a given name pertains to people of a specific gender. With this simples method, it achieves more than 95% accuracy in gender prediction based on first names.

So why build a new version? Because many Brazilian first names are not in the IBGE's data. In these cases -- that can return as many as 5~10% missings in some applications --, we cannot use [gendeBR](https://github.com/meirelesff/genderBR).

## Solution

This repo contains some exploratory code aimed to improve on the previous algorithm. By using machine learning models, we can capture patterns in first names that are good at predicting gender -- and then generalize these patterns to predict gender from unseen, novel Brazilian first names.

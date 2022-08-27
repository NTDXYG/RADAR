- Heuristic 1. The code needs to be parsed through the abstract syntax tree tool to ensure that the syntax is correct. 

- Heuristic 2. The number of sub-words of the method name is not less than 2 and the length of each sub-word is not more than 16. 

- Heuristic 3. The length of the functional description should be no more than 50 and no less than 4. 

- Heuristic 4. The length of the code should be no more than 256. 

- Heuristic 5. Remove annotation information, exception codes, and URL information from the code. 

- Heuristic 6. Unify method names in Java data to hump naming rules and unify method names in Python data to snake naming rules

By following the previous work, we randomly select 100,000 examples for training, and 2,000 examples for validation and testing, respectively.

|        | Statistics     | for     | Functional    | Description | Length     |         |
| ------ | -------------- | ------- | ------------- | ----------- | ---------- | ------- |
| ~      | Avg            | Mode    | Median        | <16         | <32        | <64     |
| Java   | 14\.25         | 8       | 11            | 69\.52%     | 93\.52%    | 99\.99% |
| Python | 17\.88         | 8       | 13            | 58\.45%     | 82\.86%    | 99\.85% |
|        | **Statistics** | **for** | **Signature** | **Length**  |            |         |
| ~      | Avg            | Mode    | Median        | <8          | <16        | <32     |
| Java   | 8\.49          | 7       | 7             | 58\.44%     | 93\.94%    | 99\.85% |
| Python | 7\.78          | 6       | 6             | 55\.48%     | 96\.92%    | 99\.98% |
|        | **Statistics** | **for** | **Method**    | **Name**    | **Length** |         |
| ~      | Avg            | Mode    | Median        | <4          | <8         | <16     |
| Java   | 2\.85          | 2       | 3             | 79\.36%     | 99\.58%    | 99\.99% |
| Python | 2\.74          | 2       | 3             | 83\.58%     | 99\.92%    | 100%    |
|        | **Statistics** | **for** | **Code**      | **Length**  |            |         |
| ~      | Avg            | Mode    | Median        | <64         | <128       | <256    |
| Java   | 40\.46         | 28      | 38            | 88\.86%     | 99\.99%    | 100%    |
| Python | 69\.44         | 42      | 63            | 50\.38%     | 92\.54%    | 100%    |
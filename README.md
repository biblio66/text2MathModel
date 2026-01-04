# text2MathModel
Text to Mathematical model using neural network (sequence-to-sequence - seq2seq model).

tensorflow library needed to run this demo.
```
python demo.py
```

**Example input text:**
```
'xssmall' has 2 GB 'ram', 2 'vcpu', costs 2.3
'xsmall' has 4 GB 'ram', 2 'vcpu', costs 3.3
'xmedium' has 8 GB 'ram', 4 'vcpu', costs 5.5
'xlarge' has 32 GB 'ram', 16 'vcpu', costs 10
'xxlarge' has 64 GB 'ram', 32 'vcpu', costs 25
at least 8 GB 'ram', 4 'vcpu'
select 1 type
```


**Output (linear programming model):**
```
Minimize
obj: 10 xlarge + 5.5 xmedium + 3.3 xsmall + 2.3 xssmall + 25 xxlarge 

Subject To
main: xlarge + xmedium + xsmall + xssmall + xxlarge =1
ram: 2 xssmall + 4 xsmall + 8 xmedium + 32 xlarge + 64 xxlarge >=8
vcpu: 2 xssmall + 2 xsmall + 4 xmedium + 16 xlarge + 32 xxlarge >=4

Binary
 xlarge
 xmedium
 xsmall
 xssmall
 xxlarge
End

solve it using any lp solver.
Example: GLPK (GNU Linear Programming Kit)
$ glpsol --lp problem.lp -o result
```

<img width="640" height="480" alt="1" src="https://github.com/user-attachments/assets/147e5fd6-4915-46d8-9771-00f69dabdbb3" />
<img width="640" height="480" alt="2" src="https://github.com/user-attachments/assets/53225290-5890-43d7-b8a0-2d3dd2ccbd97" />


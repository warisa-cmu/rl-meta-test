# Notes on LARGE3 experiments

I tried LARGE3 runs with 4 different simulations and got the following global values:
Global Value: 683.337 (3 runs, use target_solution=480, alpha_patience=10)
Global Value: 684.765 (1 run, use target_solution=600, alpha_patience=5)

Note that when using agents from T03, the result is actually worse. So it seems that transferred learning is not effective here.

# Discussion

The best solution is not optimal. This might due to wrong input parameters. From the student.

> File ก่อนหน้านี้ผมใส่ไปเป็น 2000 ครับ แต่เหมือนจะต้องใส่เป็น 240 ครับ ไม่แน่ใจกว่า Time window ของ Depot ที่รถต้องกลับมาก่อนหน้านี้กว้างไปรึเปล่าเลยทำให้คำตอบหายาก แต่ก็ไม่แน่ใจว่า พอแคบไปจะทำให้เจอ best Solution ไหมครับ 55

# Speedup for python code

1. 用set而非list进行查找
2. 用dict而非两个list进行匹配查找
3. 优先使用for循环而不是while循环
4. 用循环机制代替递归函数
5. 用缓存机制加速递归函数(@lru_cache)
6. 用numba加速Python函数
7. 使用collections.Counter加速计数
8. 使用collections.ChainMap加速字典合并
9. use np.array replace list of number
10. 使用np.ufunc代替math.func
11. 使用np.where代替if
12. 使用预分配存储代替动态扩容: 不创建空的DataFrame，预先分配存储，对所有数据赋值
13. 使用csv文件读写代替excel文件读写
14. 使用pandas多进程工具pandarallel
15. 使用dask加速dataframe
16. 应用多线程加速IO密集型任务
17. 应用多进程加速CPU密集型任务
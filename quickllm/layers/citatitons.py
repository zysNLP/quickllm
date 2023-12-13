# -*- coding: utf-8 -*- 
"""
    @Project ：quickllm 
    @File    ：citatitons.py
    @Author  ：ys
    @Time    ：2023/12/13 12:36 
"""

from moe import MoE

str_moe1 = """
    @misc{shazeer2017outrageously,
    title   = {Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer},
    author  = {Noam Shazeer and Azalia Mirhoseini and Krzysztof Maziarz and Andy Davis and Quoc Le and Geoffrey Hinton and Jeff Dean},
    year    = {2017},
    eprint  = {1701.06538},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
    }"""

str_moe2 = """
    @misc{lepikhin2020gshard,
    title   = {GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding},
    author  = {Dmitry Lepikhin and HyoukJoong Lee and Yuanzhong Xu and Dehao Chen and Orhan Firat and Yanping Huang and Maxim Krikun and Noam Shazeer and Zhifeng Chen},
    year    = {2020},
    eprint  = {2006.16668},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
    }"""

moe_citations = {
    "citations_class": MoE,
    "citations_addr1": str_moe1,
    "citations_addr2": str_moe2,
}
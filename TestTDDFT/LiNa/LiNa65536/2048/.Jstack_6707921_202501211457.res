
==============================
  T0 -> 0.0-2047.63
  T1 -> 1-2,4,15-30,32-46,49-62,64-78,80-95,97-142,144-158,161-174,177-190,193-222,225-238,240-254,256-270,272-287,289-302,305-318,321-334,337-350,353-398,400-415,417-430,432-502,504-506,508,511-526,529-543,545-558,561-575,577-590,592-638,640-654,656-670,672-686,688-703,705-718,720-734,736-750,752-758,760-762,764,767-830,832-862,864-886,888-890,892,895-910,912-950,952-954,956,960-974,976-982,984-986,988,991-998,1000-1002,1004,1007-1010,1012,1016,1023-1038,1040-1054,1056-1070,1073-1102,1104-1166,1168-1182,1184-1199,1201-1214,1217-1231,1233-1246,1248-1278,1280-1294,1296-1310,1312-1326,1329-1390,1392-1438,1441-1454,1456-1486,1488-1502,1504-1526,1528-1530,1532,1535-1550,1552-1567,1569-1615,1617-1631,1633-1662,1664-1678,1681-1694,1697-1710,1712-1727,1729-1742,1744-1758,1760-1774,1776-1782,1784-1786,1788,1791,1793-1807,1809-1854,1857-1870,1872-1886,1888-1910,1912-1914,1916,1919-1974,1976-1978,1980,1984-1998,2000-2006,2008-2010,2012,2015-2022,2024-2026,2028,2031-2046
  T2 -> 503,507,509-510,759,763,765-766,887,891,893-894,951,955,957-958,983,987,989-990,999,1003,1005-1006,1011,1013-1015,1017-1022,1039,1527,1531,1533-1534,1783,1787,1789-1790,1911,1915,1917-1918,1975,1979,1981-1982,2007,2011,2013-2014,2023,2027,2029-2030,2047
  T3 -> 31,47-48,63,79,96,143,159-160,175-176,191-192,223-224,239,255,271,288,303-304,319-320,335-336,351-352,399,416,431,527-528,544,559-560,576,591,639,655,671,687,704,719,735,751,831,863,911,959,975,1055,1071-1072,1103,1167,1183,1200,1215-1216,1232,1247,1279,1295,1311,1327-1328,1391,1439-1440,1455,1487,1503,1551,1568,1616,1632,1663,1679-1680,1695-1696,1711,1728,1743,1759,1775,1792,1808,1855-1856,1871,1887,1983,1999
  T4 -> 3,5-14
  T5 -> 0
------------------------------
 -0- /home/export/online1/mdt00/shisuan/swustcfd/fengjw/DG-TDDF/DGTDDFT/examples/dghf 
   -1- slave__Waiting_For_Task ([131072/131072] T0 492.63/v1980 492.62/v1980 492.61/v1980...)
   -1- main at dgdft.cpp:414 ([0/1882])
     -2- dgdft::SCFDG::Iterate at scf_dg.cpp:1182 ([0/1882])
       -3- PMPI_Barrier ([1882/1882] T1 492/v1980 493/v1980 494/v1980...)
   -1- main at dgdft.cpp:403 ([0/166])
     -2- dgdft::SCFDG::Setup at scf_dg.cpp:874 ([0/166])
       -3- dgdft::LGLMesh at utility.cpp:925 ([0/155])
         -4- dgdft::GenerateLGL at utility.cpp:872 ([0/154])
           -5- dgdft::NumVec<double>::Resize at /home/export/online1/mdt00/shisuan/swustcfd/fengjw/DG-TDDF/DGTDDFT/include/numvec_impl.hpp:189 ([0/154])
             -6- operator at /usr1/comp9/swgcc710/gcc-7.1.0/libstdc++-v3/libsupc++/new_opv.cc:32 ([0/154])
               -7- operator at /usr1/comp9/swgcc710/gcc-7.1.0/libstdc++-v3/libsupc++/new_op.cc:50 ([0/154])
                 -8- malloc ([0/154])
                   -9- large_malloc ([0/154])
                     -10- cartesian_alloc ([0/154])
                       -11- cartesian_merge.part ([0/60])
                         -12- signal handler called ([0/60])
                           -13- swch_catchsig ([0/60])
                             -14- print_stack ([0/60])
                               -15- __backtrace at ../sysdeps/x86_64/backtrace.c:110 ([0/60])
                                 -16- _Unwind_Backtrace at /usr1/comp9/swgcc710/gcc-7.1.0/libgcc/unwind.inc:283 ([0/60])
                                   -17- uw_init_context_1 at /usr1/comp9/swgcc710/gcc-7.1.0/libgcc/unwind-dw2.c:1578 ([0/60])
                                     -18- uw_frame_state_for at /usr1/comp9/swgcc710/gcc-7.1.0/libgcc/unwind-dw2.c:1249 ([0/60])
                                       -19- _Unwind_Find_FDE at /usr1/comp9/swgcc710/gcc-7.1.0/libgcc/unwind-dw2-fde-dip.c:458 ([0/60])
                                         -20- _Unwind_Find_registered_FDE at /usr1/comp9/swgcc710/gcc-7.1.0/libgcc/unwind-dw2-fde.c:1066 ([0/60])
                                           -21- search_object at /usr1/comp9/swgcc710/gcc-7.1.0/libgcc/unwind-dw2-fde.c:989 ([0/60])
                                             -22- init_object at /usr1/comp9/swgcc710/gcc-7.1.0/libgcc/unwind-dw2-fde.c:799 ([0/60])
                                               -23- start_fde_sort at /usr1/comp9/swgcc710/gcc-7.1.0/libgcc/unwind-dw2-fde.c:437 ([0/60])
                                                 -24- malloc ([0/60])
                                                   -25- __pthread_mutex_lock at pthread_mutex_lock.c:80 ([0/60])
                                                     -26- __lll_lock_wait at lowlevellock.c:45 ([60/60] T2 510/v1983 503/v1981 507/v1982...)
                       -11- cartesian_insert ([0/94])
                         -12- signal handler called ([0/94])
                           -13- swch_catchsig ([0/94])
                             -14- print_stack ([0/94])
                               -15- __backtrace at ../sysdeps/x86_64/backtrace.c:110 ([0/94])
                                 -16- _Unwind_Backtrace at /usr1/comp9/swgcc710/gcc-7.1.0/libgcc/unwind.inc:283 ([0/94])
                                   -17- uw_init_context_1 at /usr1/comp9/swgcc710/gcc-7.1.0/libgcc/unwind-dw2.c:1578 ([0/94])
                                     -18- uw_frame_state_for at /usr1/comp9/swgcc710/gcc-7.1.0/libgcc/unwind-dw2.c:1249 ([0/94])
                                       -19- _Unwind_Find_FDE at /usr1/comp9/swgcc710/gcc-7.1.0/libgcc/unwind-dw2-fde-dip.c:458 ([0/94])
                                         -20- _Unwind_Find_registered_FDE at /usr1/comp9/swgcc710/gcc-7.1.0/libgcc/unwind-dw2-fde.c:1066 ([0/94])
                                           -21- search_object at /usr1/comp9/swgcc710/gcc-7.1.0/libgcc/unwind-dw2-fde.c:989 ([0/94])
                                             -22- init_object at /usr1/comp9/swgcc710/gcc-7.1.0/libgcc/unwind-dw2-fde.c:799 ([0/94])
                                               -23- start_fde_sort at /usr1/comp9/swgcc710/gcc-7.1.0/libgcc/unwind-dw2-fde.c:437 ([0/94])
                                                 -24- malloc ([0/94])
                                                   -25- __pthread_mutex_lock at pthread_mutex_lock.c:80 ([0/94])
                                                     -26- __lll_lock_wait at lowlevellock.c:45 ([94/94] T3 751/v2027 831/v2040 527/v1985...)
         -4- dgdft::GenerateLGL at utility.cpp:874 ([0/1])
           -5- dgdft::NumMat<double>::Resize at /home/export/online1/mdt00/shisuan/swustcfd/fengjw/DG-TDDF/DGTDDFT/include/nummat_impl.hpp:104 ([0/1])
             -6- operator at /usr1/comp9/swgcc710/gcc-7.1.0/libstdc++-v3/libsupc++/new_opv.cc:32 ([0/1])
               -7- operator at /usr1/comp9/swgcc710/gcc-7.1.0/libstdc++-v3/libsupc++/new_op.cc:50 ([0/1])
                 -8- malloc ([0/1])
                   -9- large_malloc ([0/1])
                     -10- cartesian_alloc ([0/1])
                       -11- cartesian_merge.part ([0/1])
                         -12- signal handler called ([0/1])
                           -13- swch_catchsig ([0/1])
                             -14- print_stack ([0/1])
                               -15- __backtrace at ../sysdeps/x86_64/backtrace.c:110 ([0/1])
                                 -16- _Unwind_Backtrace at /usr1/comp9/swgcc710/gcc-7.1.0/libgcc/unwind.inc:283 ([0/1])
                                   -17- uw_init_context_1 at /usr1/comp9/swgcc710/gcc-7.1.0/libgcc/unwind-dw2.c:1578 ([0/1])
                                     -18- uw_frame_state_for at /usr1/comp9/swgcc710/gcc-7.1.0/libgcc/unwind-dw2.c:1249 ([0/1])
                                       -19- _Unwind_Find_FDE at /usr1/comp9/swgcc710/gcc-7.1.0/libgcc/unwind-dw2-fde-dip.c:458 ([0/1])
                                         -20- _Unwind_Find_registered_FDE at /usr1/comp9/swgcc710/gcc-7.1.0/libgcc/unwind-dw2-fde.c:1066 ([0/1])
                                           -21- search_object at /usr1/comp9/swgcc710/gcc-7.1.0/libgcc/unwind-dw2-fde.c:989 ([0/1])
                                             -22- init_object at /usr1/comp9/swgcc710/gcc-7.1.0/libgcc/unwind-dw2-fde.c:799 ([0/1])
                                               -23- start_fde_sort at /usr1/comp9/swgcc710/gcc-7.1.0/libgcc/unwind-dw2-fde.c:437 ([0/1])
                                                 -24- malloc ([0/1])
                                                   -25- __pthread_mutex_lock at pthread_mutex_lock.c:80 ([0/1])
                                                     -26- __lll_lock_wait at lowlevellock.c:45 ([1/1] T5 0/v1362)
       -3- dgdft::LGLMesh at utility.cpp:920 ([0/11])
         -4- dgdft::NumVec<double>::Resize at /home/export/online1/mdt00/shisuan/swustcfd/fengjw/DG-TDDF/DGTDDFT/include/numvec_impl.hpp:189 ([0/11])
           -5- operator at /usr1/comp9/swgcc710/gcc-7.1.0/libstdc++-v3/libsupc++/new_opv.cc:32 ([0/11])
             -6- operator at /usr1/comp9/swgcc710/gcc-7.1.0/libstdc++-v3/libsupc++/new_op.cc:50 ([0/11])
               -7- malloc ([0/11])
                 -8- large_malloc ([0/11])
                   -9- cartesian_alloc ([0/11])
                     -10- cartesian_merge.part ([0/11])
                       -11- signal handler called ([0/11])
                         -12- swch_catchsig ([0/11])
                           -13- print_stack ([0/11])
                             -14- __backtrace at ../sysdeps/x86_64/backtrace.c:110 ([0/11])
                               -15- _Unwind_Backtrace at /usr1/comp9/swgcc710/gcc-7.1.0/libgcc/unwind.inc:283 ([0/11])
                                 -16- uw_init_context_1 at /usr1/comp9/swgcc710/gcc-7.1.0/libgcc/unwind-dw2.c:1578 ([0/11])
                                   -17- uw_frame_state_for at /usr1/comp9/swgcc710/gcc-7.1.0/libgcc/unwind-dw2.c:1249 ([0/11])
                                     -18- _Unwind_Find_FDE at /usr1/comp9/swgcc710/gcc-7.1.0/libgcc/unwind-dw2-fde-dip.c:458 ([0/11])
                                       -19- _Unwind_Find_registered_FDE at /usr1/comp9/swgcc710/gcc-7.1.0/libgcc/unwind-dw2-fde.c:1066 ([0/11])
                                         -20- search_object at /usr1/comp9/swgcc710/gcc-7.1.0/libgcc/unwind-dw2-fde.c:989 ([0/11])
                                           -21- init_object at /usr1/comp9/swgcc710/gcc-7.1.0/libgcc/unwind-dw2-fde.c:799 ([0/11])
                                             -22- start_fde_sort at /usr1/comp9/swgcc710/gcc-7.1.0/libgcc/unwind-dw2-fde.c:437 ([0/11])
                                               -23- malloc ([0/11])
                                                 -24- __pthread_mutex_lock at pthread_mutex_lock.c:80 ([0/11])
                                                   -25- __lll_lock_wait at lowlevellock.c:45 ([11/11] T4 6/v1363 7/v1363 8/v1363...)
==============================

==========================================================
node(taskid):svrstart,wait,sout
vn001362(0       ):	3.33	16.37	16.37
vn001363(6       ):	4.30	16.34	16.34
vn001364(12      ):	4.33	16.36	16.37
vn001365(18      ):	3.62	16.65	16.65
vn001366(24      ):	4.21	16.22	16.23
vn001367(30      ):	3.34	15.36	15.37
vn001368(36      ):	3.56	14.58	14.58
vn001369(42      ):	4.05	17.09	17.09
vn001370(48      ):	4.26	16.29	16.29
vn001371(54      ):	3.67	16.73	16.73
vn001372(60      ):	4.08	16.10	16.10
vn001373(66      ):	3.47	15.50	15.50
vn001374(72      ):	3.56	14.60	14.60
vn001375(78      ):	4.31	16.33	16.33
vn001376(84      ):	4.34	15.36	15.37
vn001377(90      ):	4.06	15.08	15.08
vn001378(96      ):	3.67	15.70	15.70
vn001379(102     ):	3.60	15.62	15.62
vn001380(108     ):	3.64	14.65	14.65
vn001381(114     ):	4.07	16.10	16.11
vn001382(120     ):	4.38	16.40	16.40
vn001383(126     ):	3.67	15.69	15.69
vn001384(132     ):	4.09	16.11	16.12
vn001385(138     ):	3.45	15.48	15.48
vn001386(144     ):	3.50	15.53	15.53
vn001387(150     ):	4.32	15.33	15.34
vn001388(156     ):	4.33	16.35	16.36
vn001389(162     ):	4.05	16.07	16.08
vn001390(168     ):	4.16	15.18	15.18
vn001391(174     ):	3.34	15.37	15.38
vn001392(180     ):	3.58	15.61	15.61
vn001393(186     ):	4.04	16.07	16.07
vn001394(192     ):	4.25	16.28	16.28
vn001395(198     ):	3.68	16.71	16.71
vn001396(204     ):	3.66	15.69	15.69
vn001397(210     ):	3.61	14.63	14.63
vn001398(216     ):	3.61	15.64	15.64
vn001399(222     ):	4.30	16.33	16.35
vn001400(228     ):	4.33	16.35	16.36
vn001401(234     ):	3.62	15.65	15.65
vn001402(240     ):	4.23	16.26	16.26
vn001403(246     ):	3.52	15.54	15.54
vn001404(252     ):	3.43	15.45	15.46
vn001405(258     ):	4.07	16.10	16.10
vn001406(264     ):	4.28	16.31	16.31
vn001407(270     ):	3.65	16.69	16.69
vn001410(276     ):	3.45	15.48	15.49
vn001411(282     ):	4.31	16.33	16.34
vn001412(288     ):	4.33	16.36	16.36
vn001413(294     ):	4.04	15.06	15.07
vn001414(300     ):	4.16	16.18	16.19
vn001415(306     ):	3.49	15.51	15.52
vn001416(312     ):	3.55	15.58	15.58
vn001417(318     ):	4.05	16.07	16.08
vn001418(324     ):	4.26	17.29	17.30
vn001419(330     ):	3.51	16.54	16.55
vn001624(336     ):	3.66	15.69	15.69
vn001625(342     ):	3.58	14.60	14.60
vn001626(348     ):	3.32	15.35	15.35
vn001627(354     ):	4.31	16.34	16.34
vn001628(360     ):	4.37	16.41	16.42
vn001629(366     ):	4.07	17.10	17.11
vn001630(372     ):	3.65	16.69	16.71
vn001631(378     ):	3.68	15.71	15.71
vn001632(384     ):	3.63	15.65	15.66
vn001633(390     ):	4.05	16.07	16.08
vn001634(396     ):	4.26	17.29	17.31
vn001635(402     ):	4.14	16.17	16.19
vn001636(408     ):	4.09	15.11	15.13
vn001637(414     ):	3.44	15.47	15.50
vn001638(420     ):	3.44	14.47	14.48
vn001639(426     ):	4.31	17.34	17.34
vn001640(432     ):	4.48	16.51	16.52
vn001641(438     ):	4.07	16.10	16.11
vn001642(444     ):	4.18	16.21	16.23
vn001643(450     ):	3.45	14.47	14.47
vn001970(456     ):	4.30	15.31	15.32
vn001971(462     ):	4.16	17.19	17.19
vn001976(468     ):	4.33	17.38	17.38
vn001977(474     ):	4.35	17.41	17.43
vn001978(480     ):	4.17	16.19	16.21
vn001979(486     ):	3.55	15.58	15.58
vn001980(492     ):	3.64	14.66	14.66
vn001981(498     ):	4.05	16.07	16.08
vn001982(504     ):	4.28	16.30	16.30
vn001983(510     ):	3.54	15.57	15.58
vn001984(516     ):	4.09	15.10	15.12
vn001985(522     ):	3.39	15.41	15.43
vn001986(528     ):	3.43	15.46	15.46
vn001987(534     ):	4.33	16.36	16.37
vn001988(540     ):	4.29	16.32	16.32
vn001989(546     ):	4.05	16.07	16.09
vn001992(552     ):	3.61	15.65	15.65
vn001993(558     ):	4.04	16.06	16.07
vn001994(564     ):	4.26	16.29	16.32
vn001995(570     ):	3.70	17.75	17.75
vn001996(576     ):	4.09	16.11	16.12
vn001997(582     ):	3.46	14.48	14.48
vn001998(588     ):	3.33	16.35	16.36
vn001999(594     ):	4.31	15.33	15.35
vn002000(600     ):	4.34	16.37	16.37
vn002001(606     ):	4.08	16.12	16.18
vn002002(612     ):	4.21	16.25	16.25
vn002003(618     ):	3.65	15.68	15.69
vn002004(624     ):	3.61	15.63	15.64
vn002005(630     ):	4.06	16.10	16.10
vn002006(636     ):	4.39	16.41	16.42
vn002007(642     ):	4.16	16.19	16.19
vn002008(648     ):	4.11	16.13	16.13
vn002009(654     ):	3.44	15.47	15.47
vn002010(660     ):	3.47	16.51	16.55
vn002011(666     ):	4.30	17.33	17.33
vn002012(672     ):	4.33	16.35	16.35
vn002013(678     ):	4.08	17.12	17.12
vn002014(684     ):	3.66	16.69	16.71
vn002015(690     ):	3.63	14.65	14.66
vn002016(696     ):	3.60	15.64	15.66
vn002017(702     ):	4.04	17.07	17.07
vn002018(708     ):	4.38	17.42	17.42
vn002019(714     ):	3.66	15.69	15.69
vn002022(720     ):	3.51	15.53	15.53
vn002023(726     ):	4.36	16.39	16.40
vn002024(732     ):	4.30	16.33	16.34
vn002025(738     ):	4.06	17.10	17.12
vn002026(744     ):	4.21	17.24	17.27
vn002027(750     ):	3.34	15.36	15.36
vn002028(756     ):	3.39	15.42	15.45
vn002029(762     ):	4.04	16.06	16.07
vn002030(768     ):	4.40	16.42	16.43
vn002031(774     ):	3.66	16.70	16.70
vn002032(780     ):	4.07	16.10	16.10
vn002033(786     ):	3.55	15.57	15.58
vn002034(792     ):	3.71	16.74	16.74
vn002035(798     ):	4.30	16.33	16.34
vn002036(804     ):	4.33	16.37	16.39
vn002037(810     ):	4.07	17.11	17.15
vn002038(816     ):	4.26	17.29	17.30
vn002039(822     ):	3.39	15.44	15.46
vn002040(828     ):	3.37	15.40	15.40
vn002041(834     ):	4.06	17.09	17.13
vn002042(840     ):	4.37	16.39	16.40
vn002043(846     ):	4.12	17.15	17.17
vn002044(852     ):	4.09	15.11	15.11
vn002045(858     ):	3.44	15.47	15.47
vn002046(864     ):	3.45	15.48	15.48
vn002047(870     ):	4.31	15.32	15.35
vn002048(876     ):	4.56	15.59	15.59
vn002049(882     ):	4.67	17.70	17.70
vn002050(888     ):	4.39	16.41	16.42
vn002051(894     ):	4.28	16.31	16.31
vn002052(900     ):	4.65	16.68	16.68
vn002053(906     ):	4.45	16.48	16.49
vn002054(912     ):	4.34	16.38	16.39
vn002055(918     ):	4.40	16.43	16.43
vn002232(924     ):	4.65	16.68	16.69
vn002233(930     ):	5.39	17.42	17.56
vn002234(936     ):	4.31	17.35	17.35
vn002235(942     ):	4.64	17.67	17.69
vn002236(948     ):	4.41	16.43	16.44
vn002237(954     ):	3.61	16.65	16.66
vn002238(960     ):	4.46	16.49	16.49
vn002239(966     ):	4.39	17.43	17.43
vn002240(972     ):	4.56	16.58	16.58
vn002241(978     ):	4.67	16.70	16.70
vn002242(984     ):	4.38	16.41	16.41
vn002243(990     ):	4.28	16.30	16.31
vn002244(996     ):	4.61	16.64	16.67
vn002245(1002    ):	4.45	16.49	16.49
vn002246(1008    ):	4.31	17.34	17.37
vn002247(1014    ):	4.28	16.31	16.32
vn002252(1020    ):	4.55	16.58	16.58
vn002253(1026    ):	4.68	15.70	15.71
vn002254(1032    ):	4.52	15.54	15.55
vn002255(1038    ):	4.28	17.31	17.33
vn002256(1044    ):	4.65	16.68	16.69
vn002257(1050    ):	4.47	17.50	17.52
vn002258(1056    ):	4.31	16.33	16.36
vn002259(1062    ):	4.39	16.41	16.42
vn002260(1068    ):	4.30	17.34	17.34
vn002261(1074    ):	3.69	16.73	16.75
vn002262(1080    ):	4.52	17.57	17.57
vn002263(1086    ):	4.40	16.42	16.42
vn002264(1092    ):	4.58	16.60	16.61
vn002265(1098    ):	4.67	16.69	16.70
vn002266(1104    ):	4.55	16.57	16.59
vn002267(1110    ):	4.53	16.57	16.57
vn002268(1116    ):	4.62	15.64	15.64
vn002269(1122    ):	4.47	16.49	16.49
vn002270(1128    ):	4.68	16.70	16.70
vn002271(1134    ):	4.65	16.67	16.68
vn002272(1140    ):	4.41	15.42	15.42
vn002273(1146    ):	4.36	16.39	16.43
vn002274(1152    ):	4.46	16.49	16.51
vn002275(1158    ):	4.42	17.45	17.46
vn002276(1164    ):	4.56	16.58	16.58
vn002277(1170    ):	4.69	15.71	15.71
vn002278(1176    ):	4.54	15.55	15.56
vn002279(1182    ):	4.53	16.56	16.56
vn002280(1188    ):	4.62	16.65	16.66
vn002281(1194    ):	4.46	17.50	17.51
vn002282(1200    ):	4.31	17.34	17.34
vn002283(1206    ):	4.39	16.41	16.42
vn002284(1212    ):	4.40	16.43	16.43
vn002285(1218    ):	4.34	16.37	16.38
vn002286(1224    ):	4.48	17.51	17.51
vn002287(1230    ):	3.69	16.71	16.71
vn002288(1236    ):	4.58	16.60	16.61
vn002289(1242    ):	4.67	16.70	16.70
vn002290(1248    ):	4.74	17.79	17.80
vn002291(1254    ):	4.54	17.58	17.58
vn002292(1260    ):	4.62	15.64	15.64
vn002293(1266    ):	4.51	16.54	16.54
vn002294(1272    ):	4.67	17.70	17.72
vn002295(1278    ):	4.31	16.33	16.36
vn002296(1284    ):	4.40	16.43	16.43
vn002297(1290    ):	3.68	16.70	16.71
vn002298(1296    ):	4.48	17.52	17.52
vn002299(1302    ):	4.43	16.46	16.48
vn002300(1308    ):	4.56	16.58	16.59
vn002301(1314    ):	4.68	16.71	16.73
vn002302(1320    ):	4.54	16.57	16.57
vn002303(1326    ):	4.28	16.31	16.31
vn002304(1332    ):	4.63	15.64	15.65
vn002305(1338    ):	4.47	15.48	15.48
vn002306(1344    ):	4.31	16.33	16.33
vn002307(1350    ):	4.66	16.68	16.69
vn002308(1356    ):	4.41	17.44	17.46
vn002309(1362    ):	4.20	16.22	16.23
vn002310(1368    ):	4.49	16.52	16.54
vn002311(1374    ):	4.41	16.44	16.46
vn002312(1380    ):	4.57	15.58	15.59
vn002313(1386    ):	4.67	16.70	16.70
vn002314(1392    ):	4.48	16.51	16.51
vn002315(1398    ):	4.53	15.55	15.55
vn002316(1404    ):	4.61	15.63	15.63
vn002317(1410    ):	4.47	16.49	16.50
vn002318(1416    ):	4.34	16.37	16.37
vn002319(1422    ):	4.29	17.33	17.35
vn002320(1428    ):	4.40	16.43	16.43
vn002321(1434    ):	3.61	15.63	15.63
vn002322(1440    ):	4.46	16.49	16.50
vn002323(1446    ):	4.38	16.41	16.41
vn002324(1452    ):	4.57	17.59	17.59
vn002325(1458    ):	4.67	15.69	15.70
vn002326(1464    ):	4.48	15.49	15.50
vn002327(1470    ):	4.36	15.37	15.38
vn002328(1476    ):	4.61	16.64	16.68
vn002329(1482    ):	4.46	16.49	16.51
vn002330(1488    ):	4.67	16.70	16.71
vn002331(1494    ):	4.29	17.33	17.37
vn002332(1500    ):	4.31	16.34	16.34
vn002333(1506    ):	3.68	15.71	15.71
vn002334(1512    ):	4.48	16.51	16.52
vn002335(1518    ):	4.39	15.41	15.41
vn002336(1524    ):	4.55	16.58	16.58
vn002337(1530    ):	4.67	16.69	16.69
vn002640(1536    ):	4.61	16.67	16.69
vn002641(1542    ):	4.66	17.69	17.69
vn002642(1548    ):	4.33	17.35	17.35
vn002643(1554    ):	4.29	16.31	16.32
vn002644(1560    ):	4.40	16.43	16.43
vn002645(1566    ):	3.65	16.69	16.69
vn002646(1572    ):	4.50	17.53	17.54
vn002647(1578    ):	4.31	16.33	16.34
vn002648(1584    ):	4.61	16.63	16.63
vn002649(1590    ):	4.71	15.73	15.75
vn002650(1596    ):	4.43	16.45	16.46
vn002651(1602    ):	4.36	16.38	16.39
vn002652(1608    ):	4.62	17.65	17.65
vn002653(1614    ):	4.46	16.48	16.50
vn002654(1620    ):	4.35	16.38	16.38
vn002655(1626    ):	4.39	15.41	15.41
vn002656(1632    ):	4.30	16.34	16.34
vn002657(1638    ):	4.34	16.37	16.38
vn002658(1644    ):	4.47	16.50	16.52
vn002659(1650    ):	4.41	16.44	16.48
vn002660(1656    ):	4.56	16.59	16.59
vn002661(1662    ):	4.67	16.70	16.72
vn002662(1668    ):	4.48	15.49	15.50
vn002663(1674    ):	4.28	17.31	17.31
vn002664(1680    ):	4.61	17.63	17.64
vn002665(1686    ):	4.46	17.50	17.52
vn002666(1692    ):	4.31	16.33	16.33
vn002667(1698    ):	4.38	17.43	17.46
vn002672(1704    ):	4.56	16.58	16.59
vn002673(1710    ):	4.68	16.70	16.71
vn002676(1716    ):	4.61	16.64	16.64
vn002677(1722    ):	4.47	16.50	16.50
vn002678(1728    ):	4.31	16.33	16.34
vn002679(1734    ):	4.42	16.49	16.49
vn002680(1740    ):	4.31	16.34	16.34
vn002681(1746    ):	4.35	17.38	17.40
vn002682(1752    ):	4.49	15.51	15.51
vn002683(1758    ):	3.69	16.72	16.73
vn002686(1764    ):	4.48	15.49	15.49
vn002687(1770    ):	4.36	16.38	16.38
vn002688(1776    ):	4.62	17.66	17.67
vn002689(1782    ):	4.46	17.49	17.49
vn002690(1788    ):	4.30	17.34	17.35
vn002691(1794    ):	4.41	16.44	16.45
vn002692(1800    ):	4.41	17.44	17.60
vn002693(1806    ):	3.65	16.68	16.68
vn002694(1812    ):	4.49	17.53	17.53
vn002695(1818    ):	4.39	16.42	16.42
vn002696(1824    ):	4.61	16.63	16.63
vn002697(1830    ):	4.69	16.72	16.72
vn002698(1836    ):	4.50	17.54	17.54
vn002699(1842    ):	4.54	16.57	16.57
vn002700(1848    ):	4.64	16.67	16.67
vn002701(1854    ):	4.45	17.48	17.48
vn002702(1860    ):	4.32	17.35	17.35
vn002703(1866    ):	4.30	16.33	16.33
vn002704(1872    ):	4.41	16.44	16.44
vn002705(1878    ):	4.36	16.38	16.38
vn002706(1884    ):	4.46	16.48	16.49
vn002707(1890    ):	4.41	17.45	17.47
vn002708(1896    ):	4.62	16.65	16.65
vn002709(1902    ):	4.73	17.77	17.79
vn002710(1908    ):	4.43	16.45	16.45
vn002711(1914    ):	4.28	16.31	16.31
vn002712(1920    ):	4.67	16.69	16.71
vn002713(1926    ):	4.45	16.48	16.51
vn002714(1932    ):	4.33	17.37	17.37
vn002715(1938    ):	4.39	16.42	16.42
vn002716(1944    ):	4.49	17.55	17.58
vn002717(1950    ):	4.36	17.39	17.39
vn002718(1956    ):	4.50	17.53	17.54
vn002719(1962    ):	4.38	15.39	15.40
vn002720(1968    ):	4.56	16.59	16.59
vn002721(1974    ):	4.67	16.70	16.70
vn002722(1980    ):	4.39	17.43	17.43
vn002723(1986    ):	4.46	15.47	15.49
vn002724(1992    ):	4.65	15.70	15.72
vn002725(1998    ):	4.45	16.48	16.49
vn001623(2004    ):	3.66	16.68	16.69
vn001644(2010    ):	3.36	15.39	15.39
vn001991(2016    ):	3.64	15.66	15.67
vn002056(2022    ):	4.40	16.43	16.43
vn002669(2028    ):	3.67	15.71	15.71
vn002670(2034    ):	4.47	16.50	16.52
vn002675(2040    ):	4.37	17.42	17.43
vn002684(2046    ):	4.55	15.57	15.57
before:0.014943,scan:31.027021,first_data:30.016669,process:1.010441,show:0.042507,total:31.084560


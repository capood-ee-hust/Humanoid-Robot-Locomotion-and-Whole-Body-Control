# Mô phỏng 


## Mục lục

1. [Tổng quan](#1-tổng-quan)
2. [Cơ sở toán học](#2-cơ-sở-toán-học)
3. [Cấu trúc dự án](#3-cấu-trúc-dự-án)
4. [Cách chạy và kết quả mô phỏng](#4-cách-chạy-và-kết-quả-mô-phỏng)
5. [Yêu cầu](#5-yêu-cầu)
6. [Tài liệu tham khảo](#6-tài-liệu-tham-khảo)

---

## 1. Tổng quan

Dự án mô phỏng robot hình người Unitree G1 (29 khớp) trong bộ mô phỏng vật lý MuJoCo. Robot được điều khiển  bằng mô-men khớp.

Kịch bản thử nghiệm:

1. **Đứng cân bằng** — robot tự giữ thăng bằng khi bị đẩy. Dùng bộ điều khiển toàn thân dạng QP (QP-WBC).
2. **Đi thẳng** — bước chân → quỹ đạo DCM → động học ngược IK (phát lại động học).
3. **Đi vòng vật cản** — bộ lập bước chân A*, sau đó DCM và IK như kịch bản 2.
4. **Đi bộ động lực học, chịu được cú đẩy** — điều khiển dự báo (MPC) trên mô hình con lắc ngược tuyến tính (LIPM) chọn ZMP và dời các bước chân sắp đặt. QP-WBC đổi kết quả thành mô-men khớp.
5. **Đi bộ động lực học vòng vật cản** — đi theo kế hoạch A* của kịch bản 3 bằng MPC của kịch bản 4.

<!-- ảnh / GIF ở đây -->

---

## 2. Cơ sở toán học

### 2.1 ZMP và mô hình con lắc ngược tuyến tính (LIPM)

ZMP (trên sàn phẳng trùng với tâm áp lực CoP) là điểm đặt của tổng lực tiếp xúc:

$$p = \frac{\sum_i f_{z,i}\, r_i}{\sum_i f_{z,i}}$$

Vì $f_{z,i} \ge 0$ nên ZMP luôn nằm trong vùng đỡ (bao lồi của hai bàn chân).

LIPM giả thiết trọng tâm (CoM) chuyển động ở độ cao không đổi $z_c$ và mô-men động lượng không đổi:

$$\ddot{c} = \omega^2 (c - p), \qquad \omega = \sqrt{\frac{g}{z_c}}$$

với $c$ là vị trí ngang của trọng tâm, $p$ là ZMP. Viết lại:

$$p = c - \frac{z_c}{g}\,\ddot{c}$$

### 2.2 Thành phần phân kỳ của chuyển động (DCM)

$$\xi = c + \frac{\dot{c}}{\omega}, \qquad \dot{\xi} = \omega\,(\xi - p), \qquad \dot{c} = -\omega\,(c - \xi)$$

DCM $\xi$ là phần không ổn định của chuyển động: nó chạy ra xa ZMP. Trọng tâm chỉ việc đi theo DCM. Giữ thăng bằng tức là giữ $\xi$ không bỏ chạy. Cú đẩy ngắn $F$ trong $\Delta t$ làm DCM dịch đi

$$\Delta\xi = \frac{F\,\Delta t}{m\,\omega}$$

### 2.3 Quỹ đạo DCM từ bước chân (kịch bản 2, 3)

ZMP nằm trên từng chân trụ $z_i$ trong một bước dài $T$. DCM được tính **ngược** từ bước cuối, nơi robot đứng yên:

$$\xi^{end}_{i-1} = z_i + e^{-\omega T}\left(\xi^{end}_i - z_i\right)$$

Sau đó tích phân xuôi trọng tâm bằng $\dot{c} = \omega(\xi - c)$. Góc khớp tìm bằng IK bình phương tối thiểu có giảm chấn:

$$\Delta q = J^\top \left(J J^\top + \lambda I\right)^{-1} e$$

### 2.4 MPC có chỉnh chỗ đặt chân (kịch bản 4, 5)

Mỗi 20 ms, MPC nhìn trước $T = 1.6$ s ($N = 32$ khoảng, mỗi khoảng $\Delta = 0.05$ s). Biến quyết định là các ZMP tương lai $p_0, \dots, p_{N-1}$ và vị trí $s_1, \dots, s_M$ của tối đa $M = 3$ bước sắp đặt.

**Ràng buộc ổn định** (DCM không được phân kỳ):

$$\xi_0 = \sum_{k=0}^{N-1} a_k\, p_k + e^{-\omega T}\, \xi_T, \qquad a_k = e^{-\omega k \Delta} - e^{-\omega (k+1)\Delta}$$

**Hàm chi phí:**

$$J = w_z \sum_k \lVert p_k - r_k(s) \rVert^2 + w_d \sum_k \lVert p_k - p_{k-1} \rVert^2 + w_s \sum_m \lVert s_m - g_m \rVert^2$$

trong đó $r_k(s)$ là ZMP tham chiếu (tâm bàn chân trụ, tuyến tính theo vị trí bước chân), $g_m$ là bước chân theo kế hoạch.

**Ràng buộc:** ZMP nằm trong bàn chân ($n_i^\top (p_k - s_j) \le h_i$) hoặc trong bao lồi hai chân, và mỗi bước chân phải với tới được từ bước trước:

$$-0.12 \le e_x^\top (s_j - s_{j-1}) \le 0.32, \qquad 0.13 \le \sigma_j\, e_y^\top (s_j - s_{j-1}) \le 0.45$$

Hướng bàn chân giữ cố định nên bài toán là một QP, giải bằng **OSQP**. Chỉ áp dụng $p_0$. Giữa hai lần giải MPC, tham chiếu trọng tâm là nghiệm đóng của LIPM:

$$c(t) = p_0 + (c_0 - p_0)\cosh(\omega t) + \frac{\dot{c}_0}{\omega}\sinh(\omega t)$$

**Nhịp bước:** đồng hồ dáng đi chạy nhanh hoặc chậm hơn thời gian thật, $t_s \leftarrow t_s + \text{rate}\cdot dt$ với $\text{rate} \in [0.7, 1.6]$. Nhờ vậy sau cú đẩy robot có thể đặt chân sớm hơn hoặc đứng chân trụ lâu hơn.

### 2.5 Điều khiển toàn thân bằng QP (kịch bản 1, 4, 5)

Động lực học thân nổi của robot:

$$M(q)\,\dot{v} + h(q, v) = S^\top \tau + J_c^\top f$$

Mỗi 2 ms, WBC tìm gia tốc khớp $\dot{v}$ và lực tiếp xúc $f$:

$$\min_{\dot{v},\, f} \; \sum_i w_i \left\lVert J_i \dot{v} + \dot{J}_i v - a_i^* \right\rVert^2 + \epsilon \lVert f \rVert^2, \qquad a_i^* = \ddot{x}_i^{ref} + K_p (x_i^{ref} - x_i) + K_d (\dot{x}_i^{ref} - \dot{x}_i)$$

Các task: trọng tâm, hướng hông và thân, tư thế, chân đung đưa, mô-men động lượng ($\dot{L} = -kL$, giữ CoP thật gần ZMP mà MPC chọn).

Ràng buộc: 6 hàng đầu của phương trình động lực học (thân nổi), chân trụ không dịch chuyển, chóp ma sát $|f_x|, |f_y| \le \frac{\mu}{\sqrt{2}} f_z$, $f_z \ge 0$, và giới hạn mô-men khớp. Mô-men khớp:

$$\tau = M_a \dot{v} + h_a - J_{c,a}^\top f$$

Lớp an toàn chuyển sang giảm chấn mềm ($\tau = -k_d \dot{q}$) khi robot ngã hoặc bộ giải lỗi.

---

## 3. Sơ đồ khói hệ thống



---

## 4. Cách chạy và kết quả mô phổng 
```bash
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Tải mô hình robot:
```bash
python setup_models.py
```

Sau đó chạy một kịch bản:
```bash
cd P06_g1_balance_wbc && python interactive.py         # 1. đứng cân bằng
cd P05_footstep_dcm  && python walk_straight.py        # 2. đi thẳng
cd P05_footstep_dcm  && python run_walk_plan.py        # 3. đi vòng vật cản
cd P07_g1_mpc_walking && python walk_straight_mpc.py   # 4. đi bộ động lực học
cd P07_g1_mpc_walking && python walk_plan_mpc.py       # 5. đi bộ động lực học vòng vật cản
```

Trong cửa sổ MuJoCo: phím **8 / 9 / 6 / 7** đẩy robot về trước / sau / trái / phải, **Enter** đặt lại trạng thái ban đầu, **Space** tạm dừng.
---


<img width="990" height="770" alt="interactive" src="https://github.com/user-attachments/assets/0c6bbb17-f25c-43da-a8e6-5e56c316a8a1" />



https://github.com/user-attachments/assets/899ddedb-263b-4cb2-bacc-3fb58141618a





## 5. Yêu cầu

1. Python >= 3.10 
2. MuJoCo, OSQP, NumPy, SciPy, Matplotlib, Pillow


---

## 6. Tài liệu tham khảo

- Kajita, S. et al. (2001). *The 3D Linear Inverted Pendulum Mode: A Simple Modeling for a Biped Walking Pattern Generation.* IROS.
- Kajita, S. et al. (2003). *Biped Walking Pattern Generation by Using Preview Control of Zero-Moment Point.* ICRA.
- Pratt, J. et al. (2006). *Capture Point: A Step toward Humanoid Push Recovery.* Humanoids.
- Englsberger, J., Ott, C. & Albu-Schäffer, A. (2015). *Three-Dimensional Bipedal Walking Control Based on Divergent Component of Motion.* IEEE Transactions on Robotics.
- Wieber, P.-B. (2006). *Trajectory Free Linear Model Predictive Control for Stable Walking in the Presence of Strong Perturbations.* Humanoids.
- Herdt, A. et al. (2010). *Online Walking Motion Generation with Automatic Footstep Placement.* Advanced Robotics.
- Scianca, N. et al. (2020). *MPC for Humanoid Gait Generation: Stability and Feasibility.* IEEE Transactions on Robotics.
- Khadiv, M. et al. (2020). *Walking Control Based on Step Timing Adaptation.* IEEE Transactions on Robotics.
- Orin, D., Goswami, A. & Lee, S.-H. (2013). *Centroidal Dynamics of a Humanoid Robot.* Autonomous Robots.
- Vukobratovic, M. & Borovac, B. (2004). *Zero-Moment Point — Thirty Five Years of Its Life.* International Journal of Humanoid Robotics.
- Stellato, B. et al. (2020). *OSQP: An Operator Splitting Solver for Quadratic Programs.* Mathematical Programming Computation.
- Tài liệu MuJoCo: https://mujoco.readthedocs.io
- MuJoCo Menagerie (Unitree G1): https://github.com/google-deepmind/mujoco_menagerie

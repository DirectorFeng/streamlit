import streamlit as st
from streamlit.elements.image import image_to_url
import cv2
import numpy as np
from utils.predict import nodule_predict
from utils.computes import *
import time


colors = [(0, 0, 255), '红色',
          (0, 255, 0), '绿色',
          (255, 0, 255), '紫色',
          (255, 255, 0), '青色',
          (0, 255, 255), '黄色',
          (255, 0, 0), '蓝色']
params = []


st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_page_config(page_title="在线肺腺癌病灶检测-功能页", page_icon="💻", layout="wide")
st.sidebar.header("在线肺腺癌病灶检测-:blue[功能页]")
img_url = image_to_url("background.jpg", width=-3, clamp=False, channels='RGB', output_format='auto', image_id='')
st.markdown('''
<style>.css-fg4pbf {background-image: url(''' + img_url + ''');
    width:100%;
    height:100%;
    background-size: cover;
    background-position: center;}</style>
''', unsafe_allow_html=True)
hide_streamlit_style = """
    <style>
    footer {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
hide_streamlit_style = """
    <style>
    #footer {
                position: fixed;
                bottom: 0;
                text-align: center;
                color: black;
                font-family: Arial;
                font-size: 12px;
                letter-spacing: 1px;
            }
            
    </style>
    <div id="footer">©2023 桂林电子科技大学. All Right Reserved</div>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
upload_file = st.file_uploader("**请选择一张CT图片上传**", type=['png'],
                               help=":red[可以点击按钮上传，也可以拖拽文件上传，注意只支持PNG图片格式]")
if upload_file is not None:
    upload_flag = True
    image = np.array(bytearray(upload_file.read()))
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("./tmp/original.png", gray_img)
    img_resize = cv2.resize(gray_img, (256, 256))
    st.success('成功上传图片', icon="✅")
    with st.columns(3)[1]:
        st.image(cv2.resize(gray_img, (2048, 2048)), caption="您上传的图片")
    input_image = img_resize.reshape(1, 1, img_resize.shape[0], img_resize.shape[1])
    option = st.selectbox("**选择一种算法模型**",
                          ('Unet', 'AttentionUnet', "EfficientUnet++"))
    st.info('点击按钮即可进行病灶检测👇', icon="ℹ️")
    with st.columns(3)[1]:
        predict = st.button(":green[👉开始检测👈]", use_container_width=True)
    if predict:
        latest_iteration = st.empty()
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.05)
            my_bar.progress(percent_complete + 1)
            latest_iteration.text(f"😄AI检测中，请耐心等待。当前进度：{percent_complete + 1}%")
        result = nodule_predict(input_image=input_image, option=option)
        cv2.imwrite("./result/result.png", result)
        # time.sleep(0.3)
        # latest_iteration.text(f"AI检测中，请耐心等待。当前进度：{100}%")
        st.success('检测成功！请下滑页面查看检测结果', icon="✅")
        nodule = cv2.imread("./result/result.png", 0)
        gray_img = cv2.imread("./tmp/original.png", 0)
        # st.markdown("**检测完成！结果如下**")
        st.info('检测结果👇', icon="ℹ️")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(gray_img, caption="原CT")
        with col2:
            st.image(nodule, caption="病灶检测结果")
        with col3:
            # nodule_ = cv2.cvtColor(nodule, cv2.COLOR_GRAY2RGB)
            gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
            # nodule_params = contours_norm_compute(nodule)
            img_ = nodule.copy()
            ret, dst = cv2.threshold(nodule, 0, 255, cv2.THRESH_BINARY)
            cv2.imwrite("./result/dst.png", dst)
            contours, hierachy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            img = cv2.cvtColor(nodule, cv2.COLOR_GRAY2RGB)
            img_ = cv2.cvtColor(img_, cv2.COLOR_GRAY2RGB)
            for i in range(len(contours)):
                cv2.drawContours(img, contours[i], -1, colors[i * 2 % len(colors)], 2)
                area = round(cv2.contourArea(contours[i]), 1)
                length = round(cv2.arcLength(contours[i], True), 1)
                print(f'{colors[(i * 2 + 1) % len(colors)]}色框区域结节的面积为：{area}')
                print(f'{colors[(i * 2 + 1) % len(colors)]}色框区域结节的周长为：{length}')
                r = cv2.minAreaRect(contours[i])
                box = cv2.boxPoints(r)
                box = np.intp(box)
                cv2.drawContours(img_, [box], 0, colors[i * 2 % len(colors)], 2)
                M = cv2.moments(contours[i])
                # print(m)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                print(f'{colors[(i * 2 + 1) % len(colors)]}色框区域结节的质心为：[{cx}, {cy}]')
                diam = round(math.sqrt(4 * area / math.pi), 1)
                types = ""
                if diam < 5.0:
                    types = '微小结节'
                elif 5.0 <= diam < 10.0:
                    types = "小结节"
                elif 10.0 <= diam < 20.0:
                    types = "中型结节"
                elif 20.0 <= diam < 30.0:
                    types = "大结节"
                elif diam >= 30.0:
                    types = "肺肿块"
                print(f'{colors[(i * 2 + 1) % len(colors)]}色框区域结节的最大直径为：{diam}')
                print(f'{colors[(i * 2 + 1) % len(colors)]}色框区域结节的结节类型为：{types}')
                params.append([colors[(i * 2 + 1) % len(colors)], (cx, cy), area, diam, types])
            # st.write(params)
            df = pd.DataFrame(columns=["区域", "中心坐标/(x,y)", "面积/mm²", "最大直径/mm", "类型"])
            for i in range(len(params)):
                new_row = [params[i][0], params[i][1],
                           params[i][2], params[i][3], params[i][4]]
                df.loc[len(df)] = new_row
            cv2.imwrite("./result/contour.png", img)
            cv2.imwrite("./result/minAreaRect.png", img_)
            img_[dst == 255] = 0
            cv2.imwrite("./result/rect.png", img_)

            norm = cv2.imread("./result/rect.png")
            # time.sleep(0.5)
            # roi = cv2.addWeighted(gray_img, 0.8, norm, 1, 0)
            roi = cv2.add(norm, gray_img)
            roi = cv2.resize(roi, (2048, 2048))
            st.image(roi, caption="结果标注", channels='BGR')
        # nodule_df = contours_norm_compute(nodule)
        # st.dataframe(nodule_df, use_container_width=True)
        # report = st.button("Report")
        # if report:
            # st.experimental_rerun()
        # time.sleep(0.5)
        # tab1, tab2, tab3, tab4 = st.tabs(["结节1", "结节2", "结节3", "结节4"])
        # with tab1:
        # df.to_csv("./result/info.csv")
        # df2 = pd.read_csv("./result/info.csv")
        nodule_num = len(params)
        st.info(f'共发现了**{nodule_num}**个结节，详情请查看生成报告👇', icon="ℹ️")
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["1️⃣结节1", "2️⃣结节2", "3️⃣结节3", "4️⃣结节4", "5️⃣结节5"])
        with tab1:
            if nodule_num >= 1:
                # st.metric(label="结节区域", value=f"{params[0][0]}")
                # st.metric(label="中心坐标/(x,y)", value=f"{params[0][1]}")
                # st.metric(label="结节大小/mm²", value=f"{params[0][2]}")
                # st.metric(label="最大直径/mm", value=f"{params[0][3]}")
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric(label="区域", value=f"🔴{params[0][0]}")
                col2.metric(label="中心坐标/(x,y)", value=f"{params[0][1]}")
                col3.metric(label="面积/mm²", value=f"{params[0][2]}")
                col4.metric(label="最大直径/mm", value=f"{params[0][3]}")
                col5.metric(label="类型", value=f"{params[0][4]}")
            else:
                st.write("该CT图片中未检测到病灶")
        with tab2:
            if nodule_num >= 2:
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric(label="区域", value=f"🟢{params[1][0]}")
                col2.metric(label="中心坐标/(x,y)", value=f"{params[1][1]}")
                col3.metric(label="面积/mm²", value=f"{params[1][2]}")
                col4.metric(label="最大直径/mm", value=f"{params[1][3]}")
                col5.metric(label="类型", value=f"{params[1][4]}")
            else:
                st.write("\-")
        with tab3:
            if nodule_num >= 3:
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric(label="区域", value=f"🟣{params[2][0]}")
                col2.metric(label="中心坐标/(x,y)", value=f"{params[2][1]}")
                col3.metric(label="面积/mm²", value=f"{params[2][2]}")
                col4.metric(label="最大直径/mm", value=f"{params[2][3]}")
                col5.metric(label="类型", value=f"{params[2][4]}")
            else:
                st.write("\-")
        with tab4:
            if nodule_num >= 4:
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric(label="区域", value=f"🔵{params[3][0]}")
                col2.metric(label="中心坐标/(x,y)", value=f"{params[3][1]}")
                col3.metric(label="面积/mm²", value=f"{params[3][2]}")
                col4.metric(label="最大直径/mm", value=f"{params[3][3]}")
                col5.metric(label="类型", value=f"{params[3][4]}")
            else:
                st.write("\-")
        with tab5:
            if nodule_num == 5:
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric(label="区域", value=f"🟡{params[4][0]}")
                col2.metric(label="中心坐标/(x,y)", value=f"{params[4][1]}")
                col3.metric(label="面积/mm²", value=f"{params[4][2]}")
                col4.metric(label="最大直径/mm", value=f"{params[4][3]}")
                col5.metric(label="类型", value=f"{params[4][4]}")
            else:
                st.write("\-")

        # st.dataframe(df, use_container_width=True)
        st.write("❕注意：AI预测结果仅供参考，请以实际医生诊断情况为准！")
        report = df.to_csv()
        with st.columns(3)[1]:
            st.download_button(":green[保存报告至Excel]", report,
                               file_name="检测报告.csv",
                               use_container_width=True)

    # rerun = st.button("Rerun")
    # if rerun:
    #     st.experimental_rerun()

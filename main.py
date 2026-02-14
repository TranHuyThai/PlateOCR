import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from YOLO_OCR import get_carplate

###########################
# Session States
# 1) .image = uploaded plate Image
# 2) .attendance_df = uploaded csv of plates


# Keys
# 1) .uploader_key 
# 2) 

###########################

st.title("PlateOCR")

st.set_page_config(layout="wide")
col_scanner, col_attendance = st.columns(2)

#### FUNCTIONS ####

# Highlight function
def highlight_arrived(row):
    return ["background-color: #88E788"] * len(row) if row["arrived"] else [""] * len(row)


def mark_plate_arrived(plate_number: str):
    if st.session_state.attendance_df is None:
        st.warning("No attendance list uploaded.")
        return

    df = st.session_state.attendance_df
    plate_upper = plate_number.strip().upper()

    if plate_upper in df["plate"].str.upper().values:
        df.loc[df["plate"].str.upper() == plate_upper, "arrived"] = True
        st.session_state.attendance_df = df
        st.success(f"Plate {plate_upper} marked as arrived ✅")
    else:
        st.error(f"Plate {plate_upper} not found ❌")
    # Always recompute styled table after any update --
    # styled_df = st.session_state.attendance_df.style.apply(highlight_arrived, axis=1)
    # st.dataframe(styled_df, width="stretch")



#### STORAGE #####
# Persistent storage for uploaded image
if "image" not in st.session_state:
    st.session_state.image = None

# Persistent storage for uploaded csv attendance
if "attendance_df" not in st.session_state:
    st.session_state.attendance_df = None

#### KEYS #####
# Initialize keys
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0



#### Image uploader (will be replaced by capture) ####
with col_scanner:
    st.subheader("Carplate Scanner")
    # Initializing streamlit uploader
    uploaded_plate_img = st.file_uploader(
        "Upload a plate image to scan:",
        type=["jpg", "jpeg", "png"],
        key=st.session_state.uploader_key
    )

    # Store the uploaded image in session_state
    if uploaded_plate_img:
        st.session_state.image = Image.open(uploaded_plate_img)

    # Display the uploaded image
    if st.session_state.image:
        st.image(
            st.session_state.image,
            caption="Uploaded Image",
            width=200
        )





    #### ML application ####
    col1, col2 = st.columns([1,2]) 
    with col1:
        if st.button("Scan Plate"):
            if st.session_state.image:
                img_np = np.array(st.session_state.image)
                img_np = img_np[:, :, :3]
                st.write("Image ready for processing!")
                st.write("Scanning....")

                result = get_carplate(np.array(img_np))

                if result is None:
                    st.error("No license plate detected ❌")
                else:
                    st.success(f"Plate: {result['plate']}")
                    st.write("OCR confidence:", round(result["ocr_confidence"], 3))
                    mark_plate_arrived(result['plate'])

            else:
                st.write("No car plate detected!")
    with col2:
        if st.button("Clear image"):
            if st.session_state.image:
                st.session_state.image = None
                st.session_state.uploader_key += 1 
                st.rerun()


#### Attendance checker ####
with col_attendance:
    st.subheader("Attendance List")

    if "attendance_df" not in st.session_state:
        st.session_state.attendance_df = None

    uploaded_attendance_list = st.file_uploader(
        "Upload attendance list (CSV):",
        type=["csv"],
        key="attendance_uploader"
    )

    # Load CSV ONLY ONCE
    if uploaded_attendance_list and st.session_state.attendance_df is None:
        df = pd.read_csv(uploaded_attendance_list)
        if "arrived" not in df.columns:
            df["arrived"] = False
        st.session_state.attendance_df = df

    # Render table (always)
    if st.session_state.attendance_df is not None:
        styled_df = (
            st.session_state.attendance_df
            .style
            .apply(highlight_arrived, axis=1)
        )
        st.dataframe(styled_df, width="stretch")

    

    # # Input for marking plate arrived 
    # if st.session_state.attendance_df is not None:
    #     plate_input = st.text_input("Mark a plate as arrived:")
    #     if st.button("Update Attendance"):
    #         df = st.session_state.attendance_df
    #         if plate_input.upper() in df["plate"].str.upper().values:
    #             df.loc[df["plate"].str.upper() == plate_input.upper(), "arrived"] = True
    #             st.session_state.attendance_df = df
    #             st.success(f"Plate {plate_input} marked as arrived ✅")
    #         else:
    #             st.error(f"Plate {plate_input} not found ❌")
    

        


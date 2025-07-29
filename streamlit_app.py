import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from typing import Tuple, Optional, List
import pandas as pd
import matplotlib.pyplot as plt
import time

class JumpAnalyzer:
    def __init__(self):
        self.gravity = 9.81  # m/s² (can be adjusted based on location)
    
    def downsample_video(self, video_path: str, original_fps: int, target_fps: int) -> List[np.ndarray]:
        """Extract frames at target frame rate from original video"""
        cap = cv2.VideoCapture(video_path)
        
        # Calculate frame skip ratio
        frame_skip = original_fps // target_fps
        
        frames = []
        frame_count = 0
        
        with st.spinner(f"Downsampling video to {target_fps} FPS..."):
            progress_bar = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Keep every nth frame based on target fps
                if frame_count % frame_skip == 0:
                    frames.append(frame)
                
                frame_count += 1
                
                # Update progress bar
                if frame_count % 100 == 0:
                    progress = min(frame_count / total_frames, 1.0)
                    progress_bar.progress(progress)
        
        cap.release()
        progress_bar.progress(1.0)
        
        return frames
    
    def calculate_jump_height(self, start_frame: int, end_frame: int, fps: int) -> Tuple[float, float, dict]:
        """
        Calculate jump height using physics equations
        
        Args:
            start_frame: Frame where jump starts (takeoff)
            end_frame: Frame where jump ends (landing)
            fps: Frame rate of the analyzed video
        
        Returns:
            height_meters: Jump height in meters
            height_feet: Jump height in feet
            analysis: Dictionary with detailed analysis
        """
        # Calculate flight time
        flight_frames = end_frame - start_frame
        flight_time = flight_frames / fps
        
        # For a jump, the time to reach peak height is half the total flight time
        time_to_peak = flight_time / 2
        
        # Using kinematic equation: h = 0.5 * g * t²
        # Where t is the time to reach peak height
        height_meters = 0.5 * self.gravity * (time_to_peak ** 2)
        height_feet = height_meters * 3.28084  # Convert to feet
        
        # Additional analysis
        analysis = {
            'flight_time_seconds': flight_time,
            'flight_frames': flight_frames,
            'time_to_peak_seconds': time_to_peak,
            'takeoff_velocity_ms': self.gravity * time_to_peak,
            'takeoff_velocity_mph': (self.gravity * time_to_peak) * 2.237,
            'fps_used': fps
        }
        
        return height_meters, height_feet, analysis

def get_video_info(video_path: str) -> Tuple[int, float, int, int]:
    """Get video information"""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return frame_count, fps, width, height

def main():
    st.set_page_config(
        page_title="Jump Height Analysis",
        page_icon="🏃‍♂️",
        layout="wide"
    )
    
    st.title("🏃‍♂️ Jump Height Analysis Tool")
    st.markdown("Upload a 200fps video to analyze jump performance with precise frame-by-frame analysis.")
    
    # Initialize session state
    if 'video_path' not in st.session_state:
        st.session_state.video_path = None
    if 'frames' not in st.session_state:
        st.session_state.frames = []
    if 'current_fps' not in st.session_state:
        st.session_state.current_fps = 200
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = JumpAnalyzer()
    if 'jump_start_frame' not in st.session_state:
        st.session_state.jump_start_frame = None
    if 'jump_end_frame' not in st.session_state:
        st.session_state.jump_end_frame = None
    
    # Sidebar for controls
    st.sidebar.header("📹 Video Controls")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Video (200 FPS)",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video recorded at 200 frames per second"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
            st.session_state.video_path = video_path
        
        # Get video information
        try:
            frame_count, original_fps, width, height = get_video_info(video_path)
            
            # Display video information
            st.sidebar.subheader("📊 Video Information")
            st.sidebar.metric("Original FPS", f"{original_fps:.1f}")
            st.sidebar.metric("Total Frames", frame_count)
            st.sidebar.metric("Duration", f"{frame_count/original_fps:.2f}s")
            st.sidebar.metric("Resolution", f"{width}x{height}")
            
            # Frame rate selection
            st.sidebar.subheader("🎯 Analysis Settings")
            target_fps = st.sidebar.selectbox(
                "Target Frame Rate",
                [100, 50, 25],
                index=0,
                help="Choose the frame rate for analysis"
            )
            
            # Downsample video if fps changed or not yet processed
            if (st.session_state.current_fps != target_fps or 
                len(st.session_state.frames) == 0):
                
                st.session_state.frames = st.session_state.analyzer.downsample_video(
                    video_path, int(original_fps), target_fps
                )
                st.session_state.current_fps = target_fps
            
            if st.session_state.frames:
                total_frames = len(st.session_state.frames)
                
                # Main content area
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.subheader(f"🎬 Frame Navigation ({target_fps} FPS)")
                    
                    # Frame selection slider
                    current_frame = st.slider(
                        "Current Frame",
                        min_value=0,
                        max_value=total_frames-1,
                        value=0,
                        help="Navigate through video frames"
                    )
                    
                    # Playback controls
                    control_col1, control_col2, control_col3, control_col4, control_col5, control_col6 = st.columns(6)
                    
                    with control_col1:
                        if st.button("⏮️ First"):
                            current_frame = 0
                            st.rerun()
                    
                    with control_col2:
                        if st.button("⏪ -10"):
                            current_frame = max(0, current_frame - 10)
                            st.rerun()
                    
                    with control_col3:
                        if st.button("⏪ -1"):
                            current_frame = max(0, current_frame - 1)
                            st.rerun()
                    
                    with control_col4:
                        if st.button("⏩ +1"):
                            current_frame = min(total_frames-1, current_frame + 1)
                            st.rerun()
                    
                    with control_col5:
                        if st.button("⏩ +10"):
                            current_frame = min(total_frames-1, current_frame + 10)
                            st.rerun()
                    
                    with control_col6:
                        if st.button("⏭️ Last"):
                            current_frame = total_frames-1
                            st.rerun()
                    
                    # Display current frame
                    if current_frame < len(st.session_state.frames):
                        frame = st.session_state.frames[current_frame]
                        current_time = current_frame / target_fps
                        
                        st.image(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                            caption=f"Frame {current_frame} | Time: {current_time:.3f}s",
                            use_column_width=True
                        )
                
                with col2:
                    st.subheader("🎯 Jump Analysis")
                    
                    # Current frame info
                    st.info(f"**Current Frame:** {current_frame}\n**Time:** {current_frame/target_fps:.3f}s")
                    
                    # Jump marking controls
                    st.markdown("### Mark Jump Phases")
                    
                    if st.button("🚀 Mark Takeoff", use_container_width=True):
                        st.session_state.jump_start_frame = current_frame
                        st.success(f"Takeoff marked at frame {current_frame}")
                        st.rerun()
                    
                    if st.button("🛬 Mark Landing", use_container_width=True):
                        st.session_state.jump_end_frame = current_frame
                        st.success(f"Landing marked at frame {current_frame}")
                        st.rerun()
                    
                    # Display marked frames
                    if st.session_state.jump_start_frame is not None:
                        st.metric(
                            "Takeoff Frame", 
                            st.session_state.jump_start_frame,
                            f"{st.session_state.jump_start_frame/target_fps:.3f}s"
                        )
                    
                    if st.session_state.jump_end_frame is not None:
                        st.metric(
                            "Landing Frame", 
                            st.session_state.jump_end_frame,
                            f"{st.session_state.jump_end_frame/target_fps:.3f}s"
                        )
                    
                    # Calculate jump height
                    if (st.session_state.jump_start_frame is not None and 
                        st.session_state.jump_end_frame is not None):
                        
                        if st.session_state.jump_end_frame > st.session_state.jump_start_frame:
                            height_m, height_ft, analysis = st.session_state.analyzer.calculate_jump_height(
                                st.session_state.jump_start_frame,
                                st.session_state.jump_end_frame,
                                target_fps
                            )
                            
                            st.markdown("### 📏 Jump Results")
                            
                            # Main results
                            col_m, col_ft = st.columns(2)
                            with col_m:
                                st.metric("Height (m)", f"{height_m:.3f}")
                            with col_ft:
                                st.metric("Height (ft)", f"{height_ft:.2f}")
                            
                            # Detailed analysis
                            with st.expander("📊 Detailed Analysis"):
                                st.write(f"**Flight Time:** {analysis['flight_time_seconds']:.3f} seconds")
                                st.write(f"**Flight Frames:** {analysis['flight_frames']} frames")
                                st.write(f"**Time to Peak:** {analysis['time_to_peak_seconds']:.3f} seconds")
                                st.write(f"**Takeoff Velocity:** {analysis['takeoff_velocity_ms']:.2f} m/s")
                                st.write(f"**Takeoff Velocity:** {analysis['takeoff_velocity_mph']:.2f} mph")
                        else:
                            st.error("Landing frame must be after takeoff frame!")
                    
                    # Reset button
                    if st.button("🔄 Reset Analysis", use_container_width=True):
                        st.session_state.jump_start_frame = None
                        st.session_state.jump_end_frame = None
                        st.success("Analysis reset!")
                        st.rerun()
                
                # Export analysis
                if (st.session_state.jump_start_frame is not None and 
                    st.session_state.jump_end_frame is not None and
                    st.session_state.jump_end_frame > st.session_state.jump_start_frame):
                    
                    st.subheader("📤 Export Results")
                    
                    # Create analysis report
                    height_m, height_ft, analysis = st.session_state.analyzer.calculate_jump_height(
                        st.session_state.jump_start_frame,
                        st.session_state.jump_end_frame,
                        target_fps
                    )
                    
                    report_data = {
                        'Video File': uploaded_file.name,
                        'Analysis FPS': target_fps,
                        'Takeoff Frame': st.session_state.jump_start_frame,
                        'Landing Frame': st.session_state.jump_end_frame,
                        'Jump Height (m)': round(height_m, 3),
                        'Jump Height (ft)': round(height_ft, 2),
                        'Flight Time (s)': round(analysis['flight_time_seconds'], 3),
                        'Takeoff Velocity (m/s)': round(analysis['takeoff_velocity_ms'], 2),
                        'Takeoff Velocity (mph)': round(analysis['takeoff_velocity_mph'], 2)
                    }
                    
                    # Display summary
                    st.json(report_data)
                    
                    # Create downloadable CSV
                    df = pd.DataFrame([report_data])
                    csv = df.to_csv(index=False)
                    
                    st.download_button(
                        label="📊 Download Analysis Report (CSV)",
                        data=csv,
                        file_name=f"jump_analysis_{uploaded_file.name.split('.')[0]}.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            st.info("Please ensure the uploaded file is a valid video format.")
    
    else:
        st.info("👆 Please upload a 200fps video file to begin jump analysis")
        
        # Instructions
        with st.expander("📋 How to Use This App"):
            st.markdown("""
            ### Instructions:
            1. **Upload Video**: Choose a video file recorded at 200 FPS
            2. **Select Frame Rate**: Choose your analysis frame rate (100, 50, or 25 FPS)
            3. **Navigate Frames**: Use the slider and controls to find the jump sequence
            4. **Mark Takeoff**: Click "Mark Takeoff" when the subject's feet leave the ground
            5. **Mark Landing**: Click "Mark Landing" when the subject's feet touch the ground
            6. **View Results**: The app calculates jump height using physics equations
            7. **Export Data**: Download your analysis results as a CSV file
            
            ### Physics Method:
            - **Flight Time**: Time between takeoff and landing frames
            - **Jump Height**: Calculated using h = 0.5 × g × t²
            - Where 't' is half the flight time (time to reach peak)
            - Assumes symmetric jump trajectory
            
            ### Tips for Accurate Results:
            - Ensure clear view of takeoff and landing moments
            - Mark frames precisely at the moment of takeoff/landing
            - Use higher frame rates for more precise timing
            - Side view of jumps works best for analysis
            """)
        
        # Settings
        with st.expander("⚙️ Advanced Settings"):
            gravity = st.number_input(
                "Gravity (m/s²)",
                min_value=9.0,
                max_value=10.0,
                value=9.81,
                step=0.01,
                help="Adjust gravity constant based on location (sea level ≈ 9.81)"
            )
            
            if gravity != 9.81:
                st.session_state.analyzer.gravity = gravity
                st.info(f"Gravity set to {gravity} m/s²")

if __name__ == "__main__":
    main()

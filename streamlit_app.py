import streamlit as st
import cv2
import numpy as np
import tempfile
import os

class FrameSkipJumpAnalyzer:
    def __init__(self):
        self.gravity = 9.81
    
    def get_video_info(self, video_path: str):
        """Get basic video information"""
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return frame_count, fps, width, height
    
    def get_frame_at_position(self, video_path: str, frame_number: int):
        """Extract single frame"""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    
    def calculate_frame_skip(self, original_fps: float, target_fps: int):
        """Calculate how many frames to skip"""
        return max(1, int(original_fps / target_fps))
    
    def calculate_jump_height(self, start_frame: int, end_frame: int, original_fps: float):
        """Calculate jump height using original video timing"""
        # Flight time using original fps (real time)
        flight_time = (end_frame - start_frame) / original_fps
        
        # Time to peak (half of flight time)
        time_to_peak = flight_time / 2
        
        # Jump height calculation: h = 0.5 * g * t²
        height_meters = 0.5 * self.gravity * (time_to_peak ** 2)
        height_feet = height_meters * 3.28084
        
        return {
            'height_m': height_meters,
            'height_ft': height_feet,
            'flight_time': flight_time,
            'flight_frames': end_frame - start_frame,
            'takeoff_velocity': self.gravity * time_to_peak
        }

def main():
    st.set_page_config(
        page_title="Jump Analysis - Frame Skip",
        page_icon="🏃‍♂️",
        layout="wide"
    )
    
    st.title("🏃‍♂️ Jump Height Analysis")
    st.markdown("*Upload video → Select playback speed → Mark jump start/end → Get height*")
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = FrameSkipJumpAnalyzer()
    if 'video_loaded' not in st.session_state:
        st.session_state.video_loaded = False
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Video File",
        type=['mp4', 'avi', 'mov'],
        help="Any frame rate supported - we'll detect it automatically"
    )
    
    if uploaded_file is not None:
        # Save file temporarily
        if not st.session_state.video_loaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                st.session_state.video_path = tmp_file.name
                st.session_state.video_loaded = True
        
        try:
            # Get video info
            frame_count, original_fps, width, height = st.session_state.analyzer.get_video_info(st.session_state.video_path)
            
            # Display video information
            st.subheader("📹 Video Information")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Detected FPS", f"{original_fps:.1f}")
            with col2:
                st.metric("Total Frames", frame_count)
            with col3:
                st.metric("Duration", f"{frame_count/original_fps:.2f}s")
            with col4:
                st.metric("Resolution", f"{width}x{height}")
            
            # Frame rate selection for playback
            st.subheader("🎮 Playback Settings")
            
            # Determine available frame rates based on original fps
            available_fps = []
            if original_fps >= 200:
                available_fps = [200, 100, 50, 25]
            elif original_fps >= 100:
                available_fps = [100, 50, 25]
            elif original_fps >= 50:
                available_fps = [50, 25]
            else:
                available_fps = [25]
            
            playback_fps = st.selectbox(
                "Playback Frame Rate",
                available_fps,
                index=0,
                help="Choose how fast to step through frames"
            )
            
            # Calculate frame skip
            frame_skip = st.session_state.analyzer.calculate_frame_skip(original_fps, playback_fps)
            playback_frames = frame_count // frame_skip
            
            st.info(f"Showing every {frame_skip} frame(s) = {playback_frames} total playback frames")
            
            # Frame navigation
            st.subheader("🎬 Frame Navigation")
            
            # Convert between playback frame numbers and actual frame numbers
            playback_frame = st.slider(
                "Playback Frame",
                min_value=0,
                max_value=playback_frames-1,
                value=0,
                help="Navigate through video at selected frame rate"
            )
            
            # Calculate actual frame number
            actual_frame = playback_frame * frame_skip
            
            # Navigation controls
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                if st.button("⏮️ Start"):
                    playback_frame = 0
                    st.rerun()
            
            with col2:
                if st.button("⏪ -10"):
                    playback_frame = max(0, playback_frame - 10)
                    st.rerun()
            
            with col3:
                if st.button("⏪ -1"):
                    playback_frame = max(0, playback_frame - 1)
                    st.rerun()
            
            with col4:
                if st.button("⏩ +1"):
                    playback_frame = min(playback_frames-1, playback_frame + 1)
                    st.rerun()
            
            with col5:
                if st.button("⏩ +10"):
                    playback_frame = min(playback_frames-1, playback_frame + 10)
                    st.rerun()
            
            with col6:
                if st.button("⏭️ End"):
                    playback_frame = playback_frames-1
                    st.rerun()
            
            # Display current frame
            col1, col2 = st.columns([3, 1])
            
            with col1:
                with st.spinner("Loading frame..."):
                    frame = st.session_state.analyzer.get_frame_at_position(st.session_state.video_path, actual_frame)
                
                if frame is not None:
                    actual_time = actual_frame / original_fps
                    st.image(
                        frame,
                        caption=f"Playback Frame {playback_frame} | Actual Frame {actual_frame} | Time: {actual_time:.3f}s",
                        use_column_width=True
                    )
                else:
                    st.error("Could not load frame")
            
            with col2:
                st.subheader("🎯 Mark Jump")
                
                # Current frame info
                actual_time = actual_frame / original_fps
                st.metric("Current Time", f"{actual_time:.3f}s")
                st.metric("Actual Frame", actual_frame)
                
                # Jump marking buttons
                if st.button("🚀 JUMP START", use_container_width=True, type="primary"):
                    st.session_state.jump_start = actual_frame
                    st.session_state.jump_start_time = actual_time
                    st.success(f"Jump start marked!")
                    st.rerun()
                
                if st.button("🛬 JUMP END", use_container_width=True, type="secondary"):
                    st.session_state.jump_end = actual_frame
                    st.session_state.jump_end_time = actual_time
                    st.success(f"Jump end marked!")
                    st.rerun()
                
                # Display marked points
                if 'jump_start' in st.session_state:
                    st.write(f"**Start:** Frame {st.session_state.jump_start}")
                    st.write(f"**Start Time:** {st.session_state.jump_start_time:.3f}s")
                
                if 'jump_end' in st.session_state:
                    st.write(f"**End:** Frame {st.session_state.jump_end}")
                    st.write(f"**End Time:** {st.session_state.jump_end_time:.3f}s")
                
                # Reset button
                if st.button("🔄 Reset Marks", use_container_width=True):
                    if 'jump_start' in st.session_state:
                        del st.session_state.jump_start
                        del st.session_state.jump_start_time
                    if 'jump_end' in st.session_state:
                        del st.session_state.jump_end
                        del st.session_state.jump_end_time
                    st.success("Marks reset!")
                    st.rerun()
            
            # Calculate and display results
            if ('jump_start' in st.session_state and 
                'jump_end' in st.session_state):
                
                if st.session_state.jump_end > st.session_state.jump_start:
                    st.subheader("📏 Jump Analysis Results")
                    
                    results = st.session_state.analyzer.calculate_jump_height(
                        st.session_state.jump_start,
                        st.session_state.jump_end,
                        original_fps
                    )
                    
                    # Main results display
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Jump Height", f"{results['height_m']:.3f} m", help="Maximum vertical height reached")
                    
                    with col2:
                        st.metric("Jump Height", f"{results['height_ft']:.2f} ft", help="Maximum vertical height in feet")
                    
                    with col3:
                        st.metric("Flight Time", f"{results['flight_time']:.3f} s", help="Total time in air")
                    
                    with col4:
                        st.metric("Flight Frames", f"{results['flight_frames']}", help="Frames between takeoff and landing")
                    
                    # Detailed analysis
                    with st.expander("📊 Detailed Analysis"):
                        st.write(f"**Takeoff Velocity:** {results['takeoff_velocity']:.2f} m/s ({results['takeoff_velocity']*2.237:.2f} mph)")
                        st.write(f"**Original Video FPS:** {original_fps:.1f}")
                        st.write(f"**Playback FPS Used:** {playback_fps}")
                        st.write(f"**Frame Skip Factor:** {frame_skip}")
                        st.write(f"**Jump Start Frame:** {st.session_state.jump_start} ({st.session_state.jump_start_time:.3f}s)")
                        st.write(f"**Jump End Frame:** {st.session_state.jump_end} ({st.session_state.jump_end_time:.3f}s)")
                    
                    # Copy results
                    if st.button("📋 Copy Results"):
                        result_text = f"""Jump Analysis Results:
Height: {results['height_m']:.3f} m ({results['height_ft']:.2f} ft)
Flight Time: {results['flight_time']:.3f} seconds
Takeoff Velocity: {results['takeoff_velocity']:.2f} m/s
Video FPS: {original_fps:.1f}
File: {uploaded_file.name}"""
                        st.code(result_text, language="text")
                
                else:
                    st.error("⚠️ Jump end must be after jump start!")
        
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
    
    else:
        # Instructions when no video loaded
        st.markdown("""
        ### 📋 Instructions:
        
        1. **📤 Upload Video**: Choose any video file (any frame rate supported)
        2. **🎮 Select Playback Speed**: Choose how fast to step through frames:
           - **200 fps**: Show every frame (most precise)
           - **100 fps**: Show every 2nd frame (fast + precise)
           - **50 fps**: Show every 4th frame (good for most jumps)
           - **25 fps**: Show every 8th frame (fastest navigation)
        
        3. **🎬 Navigate**: Use slider and controls to find jump sequence
        4. **🚀 Mark Start**: Click when feet leave the ground
        5. **🛬 Mark End**: Click when feet touch the ground
        6. **📏 Get Results**: Automatic jump height calculation
        
        ### 🧮 How It Works:
        - Uses **actual video timing** for precise calculations
        - Jump height: **h = ½ × g × t²** (where t = time to peak)
        - Works with any original frame rate
        - Frame skipping only affects navigation, not accuracy
        
        ### 💡 Tips:
        - **Side view** of jumps works best
        - **Mark precisely** at moment of takeoff/landing
        - **Higher playback rates** = more precise marking
        - **Lower playback rates** = faster navigation
        """)

if __name__ == "__main__":
    main()

import streamlit as st
import cv2
import numpy as np
import tempfile
import os

class LightweightJumpAnalyzer:
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
        """Extract single frame without loading entire video"""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    
    def calculate_jump_height(self, start_frame: int, end_frame: int, original_fps: float, target_fps: int):
        """Calculate jump height with downsampled timing"""
        # Calculate effective frame numbers at target fps
        frame_skip = original_fps / target_fps
        effective_start = start_frame / frame_skip
        effective_end = end_frame / frame_skip
        
        # Flight time in downsampled frames
        flight_frames = effective_end - effective_start
        flight_time = flight_frames / target_fps
        
        # Time to peak (half of flight time)
        time_to_peak = flight_time / 2
        
        # Jump height calculation: h = 0.5 * g * t²
        height_meters = 0.5 * self.gravity * (time_to_peak ** 2)
        height_feet = height_meters * 3.28084
        
        return {
            'height_m': height_meters,
            'height_ft': height_feet,
            'flight_time': flight_time,
            'flight_frames': flight_frames,
            'takeoff_velocity': self.gravity * time_to_peak
        }

def main():
    st.set_page_config(
        page_title="Jump Analysis Lite",
        page_icon="🏃‍♂️",
        layout="centered"
    )
    
    st.title("🏃‍♂️ Jump Height Calculator")
    st.markdown("*Lightweight version - Upload 200fps video and analyze jumps*")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = LightweightJumpAnalyzer()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Video (200 FPS recommended)",
        type=['mp4', 'avi', 'mov'],
        help="Smaller files work better on Streamlit Cloud"
    )
    
    if uploaded_file is not None:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        try:
            # Get video info
            frame_count, original_fps, width, height = st.session_state.analyzer.get_video_info(video_path)
            
            # Display video info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original FPS", f"{original_fps:.1f}")
            with col2:
                st.metric("Total Frames", frame_count)
            with col3:
                st.metric("Duration", f"{frame_count/original_fps:.2f}s")
            
            # Frame rate selection
            st.subheader("Analysis Settings")
            target_fps = st.selectbox(
                "Target Frame Rate for Analysis",
                [100, 50, 25],
                index=0,
                help="Lower fps = less precise but faster processing"
            )
            
            # Calculate frame skip for display
            frame_skip = max(1, int(original_fps / target_fps))
            effective_frames = frame_count // frame_skip
            
            st.info(f"Analyzing every {frame_skip} frames = {effective_frames} total analysis frames")
            
            # Frame navigation
            st.subheader("Frame Selection")
            
            # Simple frame selector
            current_frame = st.number_input(
                "Frame Number",
                min_value=0,
                max_value=frame_count-1,
                value=0,
                step=frame_skip,
                help="Navigate to find takeoff and landing moments"
            )
            
            # Quick navigation buttons
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("⏮️ Start"):
                    current_frame = 0
                    st.rerun()
            with col2:
                if st.button("⏪ -50"):
                    current_frame = max(0, current_frame - 50*frame_skip)
                    st.rerun()
            with col3:
                if st.button("⏩ +50"):
                    current_frame = min(frame_count-1, current_frame + 50*frame_skip)
                    st.rerun()
            with col4:
                if st.button("⏭️ End"):
                    current_frame = frame_count-1
                    st.rerun()
            
            # Display current frame
            with st.spinner("Loading frame..."):
                frame = st.session_state.analyzer.get_frame_at_position(video_path, current_frame)
                
            if frame is not None:
                current_time = current_frame / original_fps
                st.image(
                    frame,
                    caption=f"Frame {current_frame} | Time: {current_time:.3f}s",
                    use_column_width=True
                )
            
            # Jump marking
            st.subheader("Mark Jump Points")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🚀 Mark as TAKEOFF", use_container_width=True):
                    st.session_state.takeoff_frame = current_frame
                    st.success(f"Takeoff: Frame {current_frame} ({current_frame/original_fps:.3f}s)")
                
                if 'takeoff_frame' in st.session_state:
                    st.write(f"**Takeoff:** Frame {st.session_state.takeoff_frame} ({st.session_state.takeoff_frame/original_fps:.3f}s)")
            
            with col2:
                if st.button("🛬 Mark as LANDING", use_container_width=True):
                    st.session_state.landing_frame = current_frame
                    st.success(f"Landing: Frame {current_frame} ({current_frame/original_fps:.3f}s)")
                
                if 'landing_frame' in st.session_state:
                    st.write(f"**Landing:** Frame {st.session_state.landing_frame} ({st.session_state.landing_frame/original_fps:.3f}s)")
            
            # Calculate results
            if ('takeoff_frame' in st.session_state and 
                'landing_frame' in st.session_state and
                st.session_state.landing_frame > st.session_state.takeoff_frame):
                
                st.subheader("📏 Jump Analysis Results")
                
                results = st.session_state.analyzer.calculate_jump_height(
                    st.session_state.takeoff_frame,
                    st.session_state.landing_frame,
                    original_fps,
                    target_fps
                )
                
                # Display main results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Jump Height", f"{results['height_m']:.3f} m")
                with col2:
                    st.metric("Jump Height", f"{results['height_ft']:.2f} ft")
                with col3:
                    st.metric("Flight Time", f"{results['flight_time']:.3f} s")
                
                # Additional details
                with st.expander("📊 Detailed Analysis"):
                    st.write(f"**Flight Frames (at {target_fps} FPS):** {results['flight_frames']:.1f}")
                    st.write(f"**Takeoff Velocity:** {results['takeoff_velocity']:.2f} m/s")
                    st.write(f"**Takeoff Velocity:** {results['takeoff_velocity']*2.237:.2f} mph")
                    st.write(f"**Analysis Frame Rate:** {target_fps} FPS")
                
                # Simple export
                if st.button("📋 Copy Results to Clipboard"):
                    result_text = f"""Jump Analysis Results:
Height: {results['height_m']:.3f} m ({results['height_ft']:.2f} ft)
Flight Time: {results['flight_time']:.3f} seconds
Takeoff Velocity: {results['takeoff_velocity']:.2f} m/s
Analysis FPS: {target_fps}
File: {uploaded_file.name}"""
                    st.code(result_text)
            
            elif ('takeoff_frame' in st.session_state and 
                  'landing_frame' in st.session_state):
                st.error("⚠️ Landing frame must be after takeoff frame!")
            
            # Reset button
            if st.button("🔄 Reset Analysis"):
                if 'takeoff_frame' in st.session_state:
                    del st.session_state.takeoff_frame
                if 'landing_frame' in st.session_state:
                    del st.session_state.landing_frame
                st.success("Analysis reset!")
                st.rerun()
        
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
        
        finally:
            # Clean up temp file
            try:
                os.unlink(video_path)
            except:
                pass
    
    else:
        # Instructions
        st.markdown("""
        ### 📋 How to Use:
        1. **Upload** a video file (200fps recommended, smaller files work better)
        2. **Choose** analysis frame rate (lower = faster processing)
        3. **Navigate** to find takeoff moment and click "Mark as TAKEOFF"
        4. **Navigate** to find landing moment and click "Mark as LANDING"
        5. **View** calculated jump height and flight time
        
        ### 💡 Tips:
        - **Smaller video files** work better on Streamlit Cloud
        - **Lower analysis frame rates** process faster
        - **Mark frames precisely** at moment feet leave/touch ground
        - **Side view** of jumps works best
        
        ### 🧮 Physics:
        Jump height calculated using: **h = 0.5 × g × t²**
        - Where **t** = time to reach peak (half of flight time)
        - Assumes symmetric vertical jump trajectory
        """)

if __name__ == "__main__":
    main()

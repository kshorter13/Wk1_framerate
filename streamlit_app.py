import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from functools import lru_cache

class OptimizedJumpAnalyzer:
    def __init__(self):
        self.gravity = 9.81
        self.max_display_width = 800  # Limit display resolution
        self.frame_cache_size = 10    # Cache only recent frames
    
    @st.cache_data
    def get_video_info(_self, video_path: str):
        """Get basic video information - cached to avoid repeated reads"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return None, None, None, None
            
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return frame_count, fps, width, height
    
    def get_display_size(self, original_width, original_height):
        """Calculate efficient display size"""
        if original_width <= self.max_display_width:
            return original_width, original_height
        
        ratio = self.max_display_width / original_width
        return int(original_width * ratio), int(original_height * ratio)
    
    @st.cache_data(max_entries=10)  # Cache recent frames
    def get_frame_at_position(_self, video_path: str, frame_number: int, display_width: int = None):
        """Extract and resize frame for display - with caching"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return None
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
            
        # Convert color space
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize for display if needed
        if display_width and frame.shape[1] > display_width:
            height, width = frame.shape[:2]
            ratio = display_width / width
            new_height = int(height * ratio)
            frame = cv2.resize(frame, (display_width, new_height), interpolation=cv2.INTER_AREA)
        
        return frame
    
    def calculate_frame_skip(self, original_fps: float, target_fps: int):
        """Calculate how many frames to skip"""
        return max(1, int(original_fps / target_fps))
    
    def calculate_jump_height(self, start_frame: int, end_frame: int, original_fps: float):
        """Calculate jump height using original video timing"""
        flight_time = (end_frame - start_frame) / original_fps
        time_to_peak = flight_time / 2
        height_meters = 0.5 * self.gravity * (time_to_peak ** 2)
        
        return {
            'height_m': height_meters,
            'flight_time': flight_time,
            'flight_frames': end_frame - start_frame,
            'takeoff_velocity': self.gravity * time_to_peak
        }

def cleanup_temp_files():
    """Clean up temporary files to save space"""
    if 'video_path' in st.session_state and os.path.exists(st.session_state.video_path):
        try:
            os.unlink(st.session_state.video_path)
        except:
            pass

def main():
    st.set_page_config(
        page_title="Jump Analysis - Optimized",
        page_icon="🏃‍♂️",
        layout="wide"
    )
    
    st.title("🏃‍♂️ Jump Height Analysis (Cloud Optimized)")
    st.markdown("*Optimized for Streamlit Cloud - reduced memory usage*")
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = OptimizedJumpAnalyzer()
    if 'video_loaded' not in st.session_state:
        st.session_state.video_loaded = False
    
    # File size warning
    st.info("💡 **Tip**: For best performance on Streamlit Cloud, use videos under 50MB and under 30 seconds")
    
    # File upload with size limit info
    uploaded_file = st.file_uploader(
        "Upload Video File (Recommended: <50MB)",
        type=['mp4', 'avi', 'mov'],
        help="Smaller files work better on Streamlit Cloud"
    )
    
    if uploaded_file is not None:
        # Check file size
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        
        if file_size_mb > 100:
            st.error(f"⚠️ File too large ({file_size_mb:.1f}MB). Please use a smaller file (<100MB)")
            return
        elif file_size_mb > 50:
            st.warning(f"⚠️ Large file ({file_size_mb:.1f}MB). May cause memory issues on Streamlit Cloud")
        
        # Save file temporarily (only once)
        if not st.session_state.video_loaded or 'video_path' not in st.session_state:
            # Clean up previous file
            cleanup_temp_files()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                st.session_state.video_path = tmp_file.name
                st.session_state.video_loaded = True
                
            # Clear frame cache when new video loaded
            st.cache_data.clear()
        
        try:
            # Get video info (cached)
            video_info = st.session_state.analyzer.get_video_info(st.session_state.video_path)
            frame_count, original_fps, width, height = video_info
            
            if frame_count is None:
                st.error("Could not read video file. Please try a different format.")
                return
            
            # Display video information
            st.subheader("📹 Video Information")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("File Size", f"{file_size_mb:.1f}MB")
            with col2:
                st.metric("Detected FPS", f"{original_fps:.1f}")
            with col3:
                st.metric("Total Frames", frame_count)
            with col4:
                st.metric("Duration", f"{frame_count/original_fps:.2f}s")
            with col5:
                st.metric("Resolution", f"{width}x{height}")
            
            # Calculate display size
            display_width, display_height = st.session_state.analyzer.get_display_size(width, height)
            if display_width < width:
                st.info(f"📱 Display optimized: {width}x{height} → {display_width}x{display_height}")
            
            # Frame rate selection for playback
            st.subheader("🎮 Playback Settings")
            
            # Determine available frame rates (limited for cloud)
            max_playback_fps = min(100, int(original_fps))  # Cap at 100 for cloud
            available_fps = []
            
            if max_playback_fps >= 100:
                available_fps = [100, 50, 25, 10]
            elif max_playback_fps >= 50:
                available_fps = [50, 25, 10]
            elif max_playback_fps >= 25:
                available_fps = [25, 10]
            else:
                available_fps = [10, 5]
            
            # Filter to only include rates <= max_playback_fps
            available_fps = [fps for fps in available_fps if fps <= max_playback_fps]
            
            playback_fps = st.selectbox(
                "Playback Frame Rate",
                available_fps,
                index=len(available_fps)//2,  # Start with middle option
                help="Lower rates = faster navigation, less memory usage"
            )
            
            # Calculate frame skip
            frame_skip = st.session_state.analyzer.calculate_frame_skip(original_fps, playback_fps)
            playback_frames = frame_count // frame_skip
            
            st.info(f"📊 Showing every {frame_skip} frame(s) = {playback_frames} playback frames | Memory optimized for cloud")
            
            # Frame navigation
            st.subheader("🎬 Frame Navigation")
            
            # Use session state for current frame to avoid resets
            if 'current_playback_frame' not in st.session_state:
                st.session_state.current_playback_frame = 0
            
            # Navigation controls first (more responsive)
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                if st.button("⏮️ Start"):
                    st.session_state.current_playback_frame = 0
            
            with col2:
                if st.button("⏪ -10"):
                    st.session_state.current_playback_frame = max(0, st.session_state.current_playback_frame - 10)
            
            with col3:
                if st.button("⏪ -1"):
                    st.session_state.current_playback_frame = max(0, st.session_state.current_playback_frame - 1)
            
            with col4:
                if st.button("⏩ +1"):
                    st.session_state.current_playback_frame = min(playback_frames-1, st.session_state.current_playback_frame + 1)
            
            with col5:
                if st.button("⏩ +10"):
                    st.session_state.current_playback_frame = min(playback_frames-1, st.session_state.current_playback_frame + 10)
            
            with col6:
                if st.button("⏭️ End"):
                    st.session_state.current_playback_frame = playback_frames-1
            
            # Slider for direct navigation
            playback_frame = st.slider(
                "Playback Frame",
                min_value=0,
                max_value=playback_frames-1,
                value=st.session_state.current_playback_frame,
                help="Navigate through video"
            )
            
            # Update session state if slider changed
            if playback_frame != st.session_state.current_playback_frame:
                st.session_state.current_playback_frame = playback_frame
            
            # Calculate actual frame number
            actual_frame = st.session_state.current_playback_frame * frame_skip
            
            # Display current frame
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Load frame (cached)
                frame = st.session_state.analyzer.get_frame_at_position(
                    st.session_state.video_path, 
                    actual_frame, 
                    display_width
                )
                
                if frame is not None:
                    actual_time = actual_frame / original_fps
                    st.image(
                        frame,
                        caption=f"Frame {st.session_state.current_playback_frame}/{playback_frames-1} | Actual Frame {actual_frame} | Time: {actual_time:.3f}s",
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
                    st.success(f"Start marked!")
                
                if st.button("🛬 JUMP END", use_container_width=True, type="secondary"):
                    st.session_state.jump_end = actual_frame
                    st.session_state.jump_end_time = actual_time
                    st.success(f"End marked!")
                
                # Display marked points
                if 'jump_start' in st.session_state:
                    st.write(f"**Start:** Frame {st.session_state.jump_start}")
                    st.write(f"**Start Time:** {st.session_state.jump_start_time:.3f}s")
                
                if 'jump_end' in st.session_state:
                    st.write(f"**End:** Frame {st.session_state.jump_end}")
                    st.write(f"**End Time:** {st.session_state.jump_end_time:.3f}s")
                
                # Reset button
                if st.button("🔄 Reset Marks", use_container_width=True):
                    for key in ['jump_start', 'jump_start_time', 'jump_end', 'jump_end_time']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.success("Marks reset!")
            
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
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Jump Height", f"{results['height_m']:.3f} m")
                    
                    with col2:
                        st.metric("Flight Time", f"{results['flight_time']:.3f} s")
                    
                    with col3:
                        st.metric("Flight Frames", f"{results['flight_frames']}")
                    
                    # Detailed analysis
                    with st.expander("📊 Detailed Analysis"):
                        st.write(f"**Takeoff Velocity:** {results['takeoff_velocity']:.2f} m/s")
                        st.write(f"**Original Video FPS:** {original_fps:.1f}")
                        st.write(f"**File Size:** {file_size_mb:.1f}MB")
                        st.write(f"**Jump Duration:** {st.session_state.jump_end_time - st.session_state.jump_start_time:.3f}s")
                
                else:
                    st.error("⚠️ Jump end must be after jump start!")
        
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            st.info("Try using a smaller video file or different format")
    
    else:
        # Instructions when no video loaded
        st.markdown("
        ### 📋 Instructions (Cloud Optimized):
        
        1. **📤 Upload Video**: 
           - **Best**: Under 50MB, under 30 seconds
           - **Max**: 100MB (may be slow)
           - Any frame rate supported
        
        2. **🎮 Playback**: Lower frame rates = faster performance
        3. **🎬 Navigate**: Use buttons or slider
        4. **🚀 Mark**: Takeoff and landing points
        5. **📏 Results**: Instant calculation
    
    # Cleanup on app restart
        if st.button("🗑️ Clear Cache & Cleanup"):
            st.cache_data.clear()
            cleanup_temp_files()
            st.success("Cache cleared!")

if __name__ == "__main__":
    main()

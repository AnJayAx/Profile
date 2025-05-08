document.addEventListener('DOMContentLoaded', function() {
    // Get elements
    const preloader = document.getElementById('preloader');
    const profileGptIcon = document.getElementById('profile-gpt');
    const appWindow = document.getElementById('app-window');
    const mainContent = document.querySelector('.main-content');
    const customCursor = document.getElementById('custom-cursor');
    
    // Explicitly position cursor at the center of the screen initially
    customCursor.style.top = '50vh';
    customCursor.style.left = '50vw';
    
    // Function to get element position
    function getElementPosition(element) {
        const rect = element.getBoundingClientRect();
        return {
            x: rect.left + rect.width / 2,
            y: rect.top + rect.height / 2
        };
    }
    
    // Set cursor animation target (center of ProfileGPT icon)
    function updateCursorTarget() {
        const iconPos = getElementPosition(profileGptIcon);
        document.documentElement.style.setProperty('--cursor-end-x', `${iconPos.x}px`);
        document.documentElement.style.setProperty('--cursor-end-y', `${iconPos.y}px`);
    }
    
    // Update cursor position on resize
    window.addEventListener('resize', updateCursorTarget);
    
    // Initialize cursor position
    setTimeout(() => {
        updateCursorTarget();
        
        // Start cursor animation sequence
        customCursor.classList.add('visible');
        
        setTimeout(() => {
            // Start moving cursor with smoother animation
            customCursor.classList.add('cursor-animate');
            
            // After cursor reaches icon, simulate click
            setTimeout(() => {
                // Add active class to simulate click on ProfileGPT icon
                profileGptIcon.classList.add('active', 'click-animation');
                
                // After click animation, show app window
                setTimeout(() => {
                    appWindow.classList.add('active');
                    
                    // Hide cursor after click
                    customCursor.style.opacity = '0';
                    
                    // After window appears, wait a bit then start transitioning to main content
                    setTimeout(() => {
                        // Make the app window expand to fill the screen (first step: expand)
                        appWindow.classList.add('fullscreen');
                        
                        // After expansion completes, adjust position (second step)
                        setTimeout(() => {
                            appWindow.classList.add('fullscreen-complete');
                            
                            // Show main content inside the window
                            setTimeout(() => {
                                // Replace window content with main content
                                mainContent.style.display = 'block';
                                mainContent.style.opacity = '0';
                                
                                // Remove the loading text and keep only the spinner
                                const loadingText = appWindow.querySelector('.loading-indicator p');
                                if (loadingText) {
                                    loadingText.style.display = 'none';
                                }
                                
                                appWindow.querySelector('.window-content').appendChild(mainContent);
                                
                                setTimeout(() => {
                                    // Fade in the main content
                                    mainContent.style.opacity = '1';
                                    mainContent.style.transition = 'opacity 0.5s ease';
                                    
                                    // Hide the loading spinner
                                    appWindow.querySelector('.loading-indicator').style.display = 'none';
                                }, 100);
                            }, 100);
                        }, 500); // Wait for the size transition to complete
                    }, 3000); // Wait 3 seconds showing the "app loading"
                }, 800);
            }, 1200); // Updated from 2000ms to 1200ms to match the CSS animation duration
        }, 500);
    }, 500);
    
    // Add click effect for manual clicking (for demo purposes)
    profileGptIcon.addEventListener('click', function() {
        this.classList.add('click-animation');
        setTimeout(() => {
            this.classList.remove('click-animation');
        }, 300);
    });
    
    // Sidebar Toggle Functionality
    const closeSidebarBtn = document.getElementById('close-sidebar-btn');
    const sidebar = document.getElementById('sidebar');
    const chatArea = document.querySelector('.chat-area');
    
    if (closeSidebarBtn && sidebar) {
        closeSidebarBtn.addEventListener('click', function() {
            sidebar.classList.toggle('collapsed');
            
            // If sidebar is collapsed, show expand button
            if (sidebar.classList.contains('collapsed')) {
                closeSidebarBtn.innerHTML = '<i class="fas fa-chevron-right"></i>';
                closeSidebarBtn.title = "Expand sidebar";
                
                // Add a mini sidebar if needed
                sidebar.style.width = '60px';
            } else {
                closeSidebarBtn.innerHTML = '<i class="fas fa-chevron-left"></i>';
                closeSidebarBtn.title = "Close sidebar";
                
                // Restore sidebar width
                sidebar.style.width = '260px';
            }
        });
    }
});

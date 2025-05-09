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
    
    // Add TextScramble class for "About Me" animation
    class TextScramble {
      constructor(el) {
        this.el = el;
        this.chars = '!<>-_\\/[]{}—=+*^?#________';
        this.update = this.update.bind(this);
      }
      
      setText(newText) {
        const oldText = this.el.innerText;
        const length = Math.max(oldText.length, newText.length);
        const promise = new Promise((resolve) => this.resolve = resolve);
        this.queue = [];
        for (let i = 0; i < length; i++) {
          const from = oldText[i] || '';
          const to = newText[i] || '';
          const start = Math.floor(Math.random() * 40);
          const end = start + Math.floor(Math.random() * 40);
          this.queue.push({ from, to, start, end });
        }
        cancelAnimationFrame(this.frameRequest);
        this.frame = 0;
        this.update();
        return promise;
      }
      
      update() {
        let output = '';
        let complete = 0;
        for (let i = 0, n = this.queue.length; i < n; i++) {
          let { from, to, start, end, char } = this.queue[i];
          if (this.frame >= end) {
            complete++;
            output += to;
          } else if (this.frame >= start) {
            if (!char || Math.random() < 0.28) {
              char = this.randomChar();
              this.queue[i].char = char;
            }
            output += `<span class="dud">${char}</span>`;
          } else {
            output += from;
          }
        }
        this.el.innerHTML = output;
        if (complete === this.queue.length) {
          this.resolve();
        } else {
          this.frameRequest = requestAnimationFrame(this.update);
          this.frame++;
        }
      }
      
      randomChar() {
        return this.chars[Math.floor(Math.random() * this.chars.length)];
      }
    }
    
    // Typing animation function
    function typeAnimation(element, speed = 30) {
        const text = element.getAttribute('data-text');
        element.classList.add('typing');
        element.textContent = '';
        
        let i = 0;
        const timer = setInterval(() => {
            if (i < text.length) {
                element.textContent += text.charAt(i);
                i++;
            } else {
                clearInterval(timer);
                element.classList.remove('typing');
                // After typing is complete, show the next message
                setTimeout(() => {
                    const profileMessage = document.querySelector('[data-message-index="3"]');
                    if (profileMessage) {
                        profileMessage.classList.remove('hidden');
                        profileMessage.classList.add('animate-fadeIn');
                    }
                }, 500); // Small delay before showing the detailed profile
            }
        }, speed);
        
        return timer; // Return timer to potentially clear it later
    }
    
    // Initialize text scramble after main content is loaded
    function initTextScramble() {
        const el = document.querySelector('.text-scramble');
        if (el) {
            const fx = new TextScramble(el);
            fx.setText(el.getAttribute('data-text'));
        }
    }
    
    // After the main content is displayed, start the message sequence
    function startMessageSequence() {
        const messages = document.querySelectorAll('[data-message-index]');
        
        // Initially hide ALL messages
        messages.forEach(msg => {
            msg.classList.add('hidden');
        });
        
        // Run text scramble first
        initTextScramble();
        
        // Sequential appearance of messages
        // First show the user message with animation
        setTimeout(() => {
            const userMessage = document.querySelector('[data-message-index="0"]');
            if (userMessage) {
                userMessage.classList.remove('hidden');
                userMessage.classList.add('animate-fadeIn');
                
                // After user message appears, show the thinking message
                setTimeout(() => {
                    const thinkingMessage = document.querySelector('[data-message-index="1"]');
                    if (thinkingMessage) {
                        thinkingMessage.classList.remove('hidden');
                        thinkingMessage.classList.add('animate-pulse');
                        
                        // Then hide thinking and show the typing animation for intro message
                        setTimeout(() => {
                            thinkingMessage.classList.add('hidden');
                            
                            const introMessage = document.querySelector('[data-message-index="2"]');
                            if (introMessage) {
                                introMessage.classList.remove('hidden');
                                const typingElement = introMessage.querySelector('.typing-animation');
                                if (typingElement) {
                                    // Start typing animation - profile message will be shown when typing completes
                                    typeAnimation(typingElement);
                                }
                            }
                        }, 2000); // Show typing after thinking
                    }
                }, 1500); // Show thinking after user message
            }
        }, 800); // Initial delay before animation starts
    }
    
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

                                    initTextScramble();
                                    
                                    // Start message sequence after content appears
                                    setTimeout(startMessageSequence, 500);
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

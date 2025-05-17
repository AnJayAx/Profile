document.addEventListener('DOMContentLoaded', function() {
    // Get the page type from the body data attribute
    const pageName = document.body.dataset.page || 'unknown';
    console.log(`Page initialized: ${pageName}`);
    
    // Check if this is a non-home page
    const isNonHomePage = document.body.classList.contains('non-home-page');
    console.log("Is non-home page:", isNonHomePage);
    
    // Handle non-home pages (no preloader) differently
    if (isNonHomePage) {
        console.log("Non-home page detected, initializing directly");
        
        // Initialize text scramble 
        initTextScramble();
        
        // Initialize sidebar toggle functionality
        initSidebar();
        
        // Start message sequence with a delay
        setTimeout(() => {
            startMessageSequence();
        }, 500);
        
        return;
    }
    
    // Initialize AutoAssess demo if on the autoassess page
    if (pageName === 'autoassess') {
        console.log("AutoAssess page detected, initializing demo");
        initAutoAssessDemo();
    }
});

// Function to initialize AutoAssess demo
function initAutoAssessDemo() {
    console.log("Initializing AutoAssess Demo");
    
    const analyzeBtn = document.getElementById('analyze-btn');
    const websiteInput = document.getElementById('website-url');
    const loadingIndicator = document.getElementById('loading-indicator');
    const analysisResults = document.getElementById('analysis-results');
    
    console.log("AutoAssess elements:", { 
        analyzeBtn: analyzeBtn, 
        websiteInput: websiteInput,
        loadingIndicator: loadingIndicator,
        analysisResults: analysisResults
    });
    
    if (!analyzeBtn || !websiteInput || !loadingIndicator || !analysisResults) {
        console.error("Could not find all necessary elements for AutoAssess demo");
        return;
    }
    
    // Set up demo site thumbnail placeholder
    const siteThumbnail = document.getElementById('site-thumbnail');
    if (siteThumbnail) {
        // Default placeholder image
        siteThumbnail.src = "https://placehold.co/600x400/121212/aaaaaa?text=Website+Preview";
    }
    
    // Explicitly bind click event
    analyzeBtn.addEventListener('click', function() {
        console.log("Analyze button clicked");
        const url = websiteInput.value.trim();
        
        if (!url) {
            // Highlight input if empty
            websiteInput.classList.add('error');
            websiteInput.style.borderColor = "#e74c3c";
            setTimeout(() => {
                websiteInput.classList.remove('error');
                websiteInput.style.borderColor = "";
            }, 800);
            return;
        }
        
        // Show loading indicator
        loadingIndicator.classList.remove('loading-hidden');
        
        // Hide results if previously shown
        analysisResults.classList.add('hidden');
        
        // Start progress animation
        let currentStep = 1;
        const totalSteps = 4;
        const progressBar = document.querySelector('.progress-bar');
        const analysisStatus = document.querySelector('.analysis-status');
        const steps = document.querySelectorAll('.step');
        
        // Initialize progress bar
        if (progressBar) progressBar.style.width = '15%';
        
        // Status messages for each step
        const statusMessages = [
            "Crawling website content...",
            "Analyzing structure and content...",
            "Calculating performance scores...",
            "Generating comprehensive report..."
        ];
        
        // Animate through steps
        const progressInterval = setInterval(() => {
            // Update current step
            updateStep(currentStep);
            
            // Move to next step
            currentStep++;
            
            if (currentStep > totalSteps) {
                clearInterval(progressInterval);
                
                // After all steps complete, show results
                setTimeout(() => {
                    loadingIndicator.classList.add('loading-hidden');
                    showAnalysisResults(url);
                }, 800);
            }
        }, 1500); // Time between steps
        
        // Function to update step visual state
        function updateStep(step) {
            // Update progress bar width
            if (progressBar) {
                const progressPercentage = (step / totalSteps) * 100;
                progressBar.style.width = `${Math.max(15, progressPercentage)}%`;
            }
            
            // Update status message
            if (analysisStatus) {
                analysisStatus.textContent = statusMessages[step - 1];
            }
            
            // Update step indicators
            steps.forEach((stepEl, index) => {
                if (index + 1 < step) {
                    stepEl.classList.remove('active');
                    stepEl.classList.add('completed');
                } else if (index + 1 === step) {
                    stepEl.classList.add('active');
                    stepEl.classList.remove('completed');
                } else {
                    stepEl.classList.remove('active', 'completed');
                }
            });
        }
    });
    
    // Function to display analysis results
    function showAnalysisResults(url) {
        console.log("Showing analysis results for:", url);
        
        // Populate results with website info
        const siteTitle = document.getElementById('site-title');
        const siteUrl = document.getElementById('site-url');
        const siteDomain = document.getElementById('site-domain');
        
        // Extract domain from URL
        let domain = url.replace(/^https?:\/\//, '').split('/')[0];
        
        if (siteTitle) siteTitle.textContent = `${domain} Analysis Report`;
        if (siteUrl) siteUrl.textContent = url;
        if (siteDomain) siteDomain.textContent = domain;
        
        // Update thumbnail if possible
        const siteThumbnail = document.getElementById('site-thumbnail');
        if (siteThumbnail) {
            // In a real application, this would be an actual screenshot
            // For demo purposes, we'll use a placeholder with the domain name
            siteThumbnail.src = `https://placehold.co/600x400/121212/aaaaaa?text=${domain}`;
        }
        
        // Show results
        analysisResults.classList.remove('hidden');
        
        // Set up tab navigation
        const tabs = document.querySelectorAll('.tab');
        tabs.forEach(tab => {
            tab.addEventListener('click', function() {
                // Remove active class from all tabs
                tabs.forEach(t => t.classList.remove('active'));
                
                // Add active class to clicked tab
                this.classList.add('active');
                
                // Hide all tab content
                const tabContents = document.querySelectorAll('.tab-content');
                tabContents.forEach(content => content.classList.remove('active'));
                
                // Show corresponding content
                const tabId = this.getAttribute('data-tab');
                const activeContent = document.querySelector(`.tab-content[data-content="${tabId}"]`);
                if (activeContent) activeContent.classList.add('active');
            });
        });
    }
}

// Sidebar Toggle Functionality
function initSidebar() {
    const closeSidebarBtn = document.getElementById('close-sidebar-btn');
    const sidebar = document.getElementById('sidebar');
    
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
}

// Typing animation function - fixed syntax and simplified
function typeAnimation(element, options = {}) {
    if (!element) return Promise.resolve();
    
    // Default options
    const defaults = {
        text: element.getAttribute('data-text') || element.textContent,
        speed: 30,
        startDelay: 0,
        cursor: true,
        cursorChar: '|',
        onComplete: null
    };
    
    // Merge options with defaults
    const config = {...defaults, ...options};
    const text = config.text;
    
    // Clear element and add typing class
    element.textContent = '';
    element.classList.add('typing');
    
    // Add cursor if specified
    if (config.cursor) {
        element.style.borderRight = `2px solid #10a37f`;
    }
    
    return new Promise(resolve => {
        // Handle start delay
        setTimeout(() => {
            let i = 0;
            const timer = setInterval(() => {
                if (i < text.length) {
                    element.textContent += text.charAt(i);
                    i++;
                } else {
                    clearInterval(timer);
                    element.classList.remove('typing');
                    
                    // Keep cursor if specified
                    if (!config.cursor) {
                        element.style.borderRight = 'none';
                    } else {
                        // Add blinking cursor class
                        element.classList.add('blinking-cursor');
                    }
                    
                    // Execute completion callback if provided
                    if (typeof config.onComplete === 'function') {
                        config.onComplete(element);
                    }
                    
                    resolve(element);
                }
            }, config.speed);
        }, config.startDelay);
    });
}

// Utility function to apply typing animation
function applyTypingAnimation(selector, options = {}) {
    const elements = document.querySelectorAll(selector);
    const animations = [];
    
    elements.forEach(element => {
        animations.push(typeAnimation(element, options));
    });
    
    return Promise.all(animations);
}

// Message sequence function for content pages
async function startMessageSequence() {
    console.log("Starting message sequence...");
    const messages = document.querySelectorAll('[data-message-index]');
    if (!messages || messages.length === 0) {
        console.error("No messages found with data-message-index attribute");
        return;
    }
    
    console.log(`Found ${messages.length} messages to animate`);
    
    // Initially hide ALL messages
    messages.forEach(msg => {
        msg.classList.add('hidden');
    });
    
    // Get all messages and sort them by index
    const sortedMessages = Array.from(messages).sort((a, b) => {
        return parseInt(a.dataset.messageIndex) - parseInt(b.dataset.messageIndex);
    });
    
    // Process messages sequentially
    for (let i = 0; i < sortedMessages.length; i++) {
        const msg = sortedMessages[i];
        const index = parseInt(msg.dataset.messageIndex);
        console.log(`Processing message ${index} (${msg.classList})`);
        
        // For user messages
        if (msg.classList.contains('user-message')) {
            await new Promise(resolve => {
                setTimeout(() => {
                    msg.classList.remove('hidden');
                    msg.classList.add('animate-fadeIn');
                    console.log(`Showing user message ${index}`);
                    resolve();
                }, 800);
            });
            
            // Add 1 second delay after user message before Edson message
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
        // For thinking messages
        else if (msg.classList.contains('thinking-message')) {
            await new Promise(resolve => {
                setTimeout(() => {
                    msg.classList.remove('hidden');
                    msg.classList.add('animate-pulse');
                    console.log(`Showing thinking message ${index}`);
                    
                    // Show the thinking message for a while before hiding it
                    setTimeout(() => {
                        msg.classList.add('hidden');
                        msg.classList.remove('animate-pulse');
                        resolve();
                    }, 2000);
                }, 300); // Reduced delay for thinking message
            });
        }
        // For Edson messages
        else if (msg.classList.contains('edson-message')) {
            msg.classList.remove('hidden');
            console.log(`Showing Edson message ${index}`);
            
            // Check for typing animation
            const typingElement = msg.querySelector('.typing-animation');
            if (typingElement) {
                // Perform typing animation
                console.log(`Starting typing animation for message ${index}`);
                await typeAnimation(typingElement);
                
                // Small delay after typing completes
                await new Promise(resolve => setTimeout(resolve, 500));
            } else {
                // If no typing animation, just show the message
                msg.classList.add('animate-fadeIn');
                await new Promise(resolve => setTimeout(resolve, 800));
            }
        }
    }
    
    console.log("Message sequence completed");
}

// Initialize text scramble for page title
function initTextScramble() {
    console.log("Initializing text scramble");
    const el = document.querySelector('.text-scramble');
    if (el) {
        console.log("Text scramble element found, initializing");
        const fx = new TextScramble(el);
        fx.setText(el.getAttribute('data-text') || el.textContent);
    } else {
        console.log("No text scramble element found");
    }
}

// Class for text scramble animation
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

// Attach to window for global use
window.typeAnimation = typeAnimation;
window.applyTypingAnimation = applyTypingAnimation;

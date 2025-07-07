// ===== VoxelInsight Landing Page JavaScript =====

// Initialize Supabase client
const SUPABASE_URL = 'https://crbazohhljdegpvakwff.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNyYmF6b2hobGpkZWdwdmFrd2ZmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE4NDIzNTUsImV4cCI6MjA2NzQxODM1NX0.BlM0_YwvGnBb-slNTPB7hRUsvcaHfeCCfNXzKdR4VVY';

// Initialize Supabase client (you'll need to replace with your actual credentials)
let supabase;
try {
    supabase = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
} catch (error) {
    console.log('Supabase not configured yet. Please add your credentials.');
}

// Global variables
let currentAuthMode = 'signin';

// DOM Elements
const authModal = document.getElementById('authModal');
const authTitle = document.getElementById('authTitle');
const authForm = document.getElementById('authForm');
const authSubmit = document.getElementById('authSubmit');
const authMessage = document.getElementById('authMessage');
const authTabs = document.querySelectorAll('.auth-tab');

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Mobile menu toggle
function toggleMobileMenu() {
    const navLinks = document.querySelector('.nav-links');
    const mobileToggle = document.querySelector('.nav-mobile-toggle');
    
    navLinks.classList.toggle('active');
    mobileToggle.classList.toggle('active');
}

// Scroll to features section
function scrollToFeatures() {
    const featuresSection = document.getElementById('features');
    featuresSection.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
    });
}

// Contact sales function
function contactSales() {
    window.open('mailto:sales@voxelinsight.com?subject=Enterprise%20Inquiry', '_blank');
}

// Authentication Modal Functions
function showAuthModal(mode = 'signin') {
    currentAuthMode = mode;
    updateAuthModal();
    authModal.style.display = 'block';
    document.body.style.overflow = 'hidden';
    
    // Add entrance animation
    setTimeout(() => {
        authModal.style.opacity = '1';
    }, 10);
}

function closeAuthModal() {
    authModal.style.opacity = '0';
    setTimeout(() => {
        authModal.style.display = 'none';
        document.body.style.overflow = 'auto';
        clearAuthMessage();
    }, 300);
}

function switchAuthTab(mode) {
    currentAuthMode = mode;
    updateAuthModal();
    clearAuthMessage();
}

function updateAuthModal() {
    // Update title and button text
    if (currentAuthMode === 'signin') {
        authTitle.textContent = 'Sign In to VoxelInsight';
        authSubmit.textContent = 'Sign In';
    } else {
        authTitle.textContent = 'Create Your Account';
        authSubmit.textContent = 'Sign Up';
    }
    
    // Update tab states
    authTabs.forEach(tab => {
        tab.classList.remove('active');
        if (tab.textContent.toLowerCase().includes(currentAuthMode)) {
            tab.classList.add('active');
        }
    });
}

function clearAuthMessage() {
    authMessage.textContent = '';
    authMessage.className = 'auth-message';
}

function showAuthMessage(message, type = 'error') {
    authMessage.textContent = message;
    authMessage.className = `auth-message ${type}`;
}

// Form submission handler
authForm.addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    
    if (!email || !password) {
        showAuthMessage('Please fill in all fields');
        return;
    }
    
    try {
        authSubmit.disabled = true;
        authSubmit.textContent = 'Processing...';
        
        if (currentAuthMode === 'signin') {
            await signIn(email, password);
        } else {
            await signUp(email, password);
        }
    } catch (error) {
        showAuthMessage(error.message);
    } finally {
        authSubmit.disabled = false;
        authSubmit.textContent = currentAuthMode === 'signin' ? 'Sign In' : 'Sign Up';
    }
});

// Server Authentication Functions
async function signIn(email, password) {
    try {
        const response = await fetch('/api/auth/signin', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ email, password })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Sign in failed');
        }
        
        showAuthMessage('Successfully signed in! Redirecting...', 'success');
        setTimeout(() => {
            window.location.href = '/chat';
        }, 1500);
    } catch (error) {
        throw new Error(error.message);
    }
}

async function signUp(email, password) {
    try {
        const response = await fetch('/api/auth/signup', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ email, password })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Sign up failed');
        }
        
        showAuthMessage('Account created successfully! Redirecting to chat...', 'success');
        setTimeout(() => {
            window.location.href = '/chat';
        }, 1500);
    } catch (error) {
        throw new Error(error.message);
    }
}

async function signInWithGoogle() {
    if (!supabase) {
        showAuthMessage('Authentication not configured. Please contact support.');
        return;
    }
    
    const { data, error } = await supabase.auth.signInWithOAuth({
        provider: 'google',
        options: {
            redirectTo: window.location.origin + '/chat'
        }
    });
    
    if (error) {
        showAuthMessage(error.message);
    }
}

// Scroll Animation Observer
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const scrollObserver = new IntersectionObserver(function(entries) {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('visible');
        }
    });
}, observerOptions);

// Observe all elements with scroll animations
function observeScrollAnimations() {
    const animatedElements = document.querySelectorAll('.animate-on-scroll, .animate-on-scroll-delay');
    animatedElements.forEach(el => {
        scrollObserver.observe(el);
    });
}

// Close modal when clicking outside
window.addEventListener('click', function(e) {
    if (e.target === authModal) {
        closeAuthModal();
    }
});

// Close modal with Escape key
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape' && authModal.style.display === 'block') {
        closeAuthModal();
    }
});

// Navbar scroll effect with improved performance
let ticking = false;
function updateNavbar() {
    const navbar = document.querySelector('.navbar');
    if (window.scrollY > 50) {
        navbar.style.background = 'rgba(255, 255, 255, 0.98)';
        navbar.style.boxShadow = '0 2px 20px rgba(0, 0, 0, 0.1)';
    } else {
        navbar.style.background = 'rgba(255, 255, 255, 0.95)';
        navbar.style.boxShadow = 'none';
    }
    ticking = false;
}

window.addEventListener('scroll', function() {
    if (!ticking) {
        requestAnimationFrame(updateNavbar);
        ticking = true;
    }
});

// Parallax effect for hero section
function updateParallax() {
    const scrolled = window.pageYOffset;
    const hero = document.querySelector('.hero');
    const heroContent = document.querySelector('.hero-content');
    
    if (hero && heroContent) {
        const rate = scrolled * -0.5;
        heroContent.style.transform = `translateY(${rate}px)`;
    }
}

window.addEventListener('scroll', function() {
    if (window.scrollY < window.innerHeight) {
        updateParallax();
    }
});

// Loading animation
window.addEventListener('load', function() {
    document.body.style.opacity = '1';
    
    // Add subtle entrance animations to elements
    const elements = document.querySelectorAll('.feature-card, .step, .about-stat');
    elements.forEach((el, index) => {
        setTimeout(() => {
            el.style.opacity = '1';
            el.style.transform = 'translateY(0)';
        }, index * 100);
    });
});

// Initialize page
document.addEventListener('DOMContentLoaded', function() {
    // Add loading state
    document.body.style.opacity = '0';
    document.body.style.transition = 'opacity 0.5s ease';
    
    // Initialize scroll animations
    observeScrollAnimations();
    
    // Check if user is already authenticated by checking server session
    fetch('/api/auth/user')
        .then(response => {
            if (response.ok) {
                // User is already signed in, redirect to chat
                window.location.href = '/chat';
            }
        })
        .catch(error => {
            // User not authenticated, stay on landing page
            console.log('User not authenticated');
        });
    
    // Add entrance animations to hero elements
    setTimeout(() => {
        const heroElements = document.querySelectorAll('.hero-title, .hero-subtitle, .hero-cta');
        heroElements.forEach((el, index) => {
            setTimeout(() => {
                el.style.opacity = '1';
                el.style.transform = 'translateY(0)';
            }, index * 200);
        });
    }, 100);
    
    // Add hover effects to feature cards
    const featureCards = document.querySelectorAll('.feature-card');
    featureCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-8px) scale(1.02)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });
    
    // Add click effects to buttons
    const buttons = document.querySelectorAll('.btn-primary, .btn-secondary');
    buttons.forEach(button => {
        button.addEventListener('click', function(e) {
            // Create ripple effect
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;
            
            ripple.style.width = ripple.style.height = size + 'px';
            ripple.style.left = x + 'px';
            ripple.style.top = y + 'px';
            ripple.classList.add('ripple');
            
            this.appendChild(ripple);
            
            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    });
});

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Performance optimization: Debounced scroll handler
const debouncedScrollHandler = debounce(function() {
    // Add any scroll-based functionality here
}, 10);

window.addEventListener('scroll', debouncedScrollHandler);

// Error handling
window.addEventListener('error', function(e) {
    console.error('JavaScript error:', e.error);
});

// Service Worker registration (for PWA capabilities)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/sw.js')
            .then(function(registration) {
                console.log('SW registered: ', registration);
            })
            .catch(function(registrationError) {
                console.log('SW registration failed: ', registrationError);
            });
    });
}

// Add CSS for ripple effect
const style = document.createElement('style');
style.textContent = `
    .btn-primary, .btn-secondary {
        position: relative;
        overflow: hidden;
    }
    
    .ripple {
        position: absolute;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: scale(0);
        animation: ripple-animation 0.6s linear;
        pointer-events: none;
    }
    
    @keyframes ripple-animation {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }
    
    .feature-card {
        opacity: 0;
        transform: translateY(30px);
        transition: all 0.6s ease;
    }
    
    .feature-card.visible {
        opacity: 1;
        transform: translateY(0);
    }
`;
document.head.appendChild(style); 
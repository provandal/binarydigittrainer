# Overview

This is a comprehensive educational Binary Digit Trainer that teaches neural network fundamentals through step-by-step visualization. Users draw binary digits (0 or 1) on a 9×9 pixel canvas with direct binary pixels. The neural network architecture is 81→24→2 (81 input neurons for pixels, 24 hidden neurons, 2 output neurons for digits 0 and 1). The application demonstrates the complete training cycle: forward pass, loss calculation, and backpropagation with real-time weight and activation updates.

# User Preferences

Preferred communication style: Simple, everyday language.

**Data Protection Policy**: Never reset the database or delete entries without explicit user discussion and approval first. User training data is valuable and must be preserved.

# System Architecture

## Frontend Architecture
- **Framework**: React 18 with TypeScript for type safety
- **Styling**: Tailwind CSS with shadcn/ui component library for consistent UI design
- **Routing**: Wouter for lightweight client-side routing
- **State Management**: React hooks and TanStack Query for server state management
- **Build Tool**: Vite for fast development and optimized production builds

## Backend Architecture
- **Runtime**: Node.js with Express.js framework
- **Development Setup**: Uses tsx for TypeScript execution in development
- **API Structure**: RESTful API endpoints prefixed with `/api`
- **Storage Layer**: Abstract storage interface with in-memory implementation for development
- **Session Management**: Prepared for PostgreSQL session storage with connect-pg-simple

## Component Design Patterns
- **Composition Pattern**: Modular UI components for different aspects of neural network training
- **Hook-based Logic**: Custom hooks like `useNeuralNetwork` encapsulate complex state logic
- **Simulation Architecture**: Client-side neural network simulation for educational purposes
- **Real-time Updates**: Live visualization of training progress and network activations

## Key Features
- **Interactive Drawing Canvas**: 9×9 pixel grid with direct binary pixel values for high-resolution drawing
- **Binary Pixel Values**: Each pixel value is 0 (white) or 1 (black) for direct neural network input
- **Step-by-Step Training**: 6-stage training cycle with clear visualization of each step
- **Neural Network Architecture**: Scaled 81→24→2 network for improved binary digit classification (0 vs 1)
- **Weight Visualization**: Real-time display of connection weights with color coding
- **Database Persistence**: PostgreSQL storage for training examples with full CRUD operations
- **Automated Training**: "Run to Next Sample" button cycles through all training examples automatically
- **Inference Mode**: Real-time prediction mode where users draw digits and get instant predictions
- **Dataset Editor**: Comprehensive editor for creating, editing, and managing training examples
- **Checkpoint System**: Full model state export/import with comprehensive metadata
- **Learning Rate Decay**: Exponential decay scheduler with visual feedback
- **Activation Explorer**: Interactive 9×9 heatmap visualization of neuron weight templates with time scrubbing
- **Top Contributors Analysis**: Output neuron view with excitatory/inhibitory classification showing most influential hidden neurons with clickable mini thumbnails
- **Unified Color Schemes**: Consistent color-blind friendly options (Blue/Orange, Green/Purple, High contrast) across all visualizations
- **Educational Focus**: Designed for learning neural network fundamentals, not performance

## Database Schema
- **Users Table**: Basic user management with username/password authentication
- **Schema Validation**: Drizzle-zod integration for type-safe database operations
- **Migration Support**: Drizzle-kit for database schema migrations

# External Dependencies

## UI and Styling
- **shadcn/ui**: Complete component library built on Radix UI primitives
- **Tailwind CSS**: Utility-first CSS framework with custom design tokens
- **Radix UI**: Accessible, unstyled UI primitives for complex components
- **Lucide React**: Icon library for consistent iconography

## Data Management
- **TanStack Query**: Server state management with caching and synchronization
- **React Hook Form**: Form handling with validation support
- **Wouter**: Lightweight routing solution for single-page application navigation

## Database and ORM
- **Drizzle ORM**: Type-safe PostgreSQL ORM with migration support
- **Neon Database**: Serverless PostgreSQL database provider
- **Zod**: Runtime type validation for API requests and database schemas

## Development Tools
- **Vite**: Fast build tool with HMR and optimized bundling
- **TypeScript**: Static type checking for enhanced developer experience
- **ESBuild**: Fast JavaScript bundler for production builds
- **Replit Integration**: Development environment optimizations for Replit platform
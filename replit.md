# Overview

This is a simplified educational Binary Digit Trainer that teaches neural network fundamentals through step-by-step visualization. Users draw binary digits (0 or 1) on a 3×3 pixel canvas where each pixel is a 3×3 grid of sub-pixels. The neural network architecture is 9→4→2 (9 input neurons for pixels, 4 hidden neurons, 2 output neurons for digits 0 and 1). The application demonstrates the complete training cycle: forward pass, loss calculation, and backpropagation with real-time weight and activation updates.

# User Preferences

Preferred communication style: Simple, everyday language.

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
- **Interactive Drawing Canvas**: 3×3 pixel grid where each pixel is a 3×3 sub-pixel grid for realistic drawing
- **Continuous Pixel Values**: Each pixel value ranges from 0-1 based on filled sub-pixels (e.g., 3/9 = 0.33)
- **Step-by-Step Training**: 6-stage training cycle with clear visualization of each step
- **Neural Network Architecture**: Simple 9→4→2 network for binary digit classification (0 vs 1)
- **Weight Visualization**: Real-time display of connection weights with color coding
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
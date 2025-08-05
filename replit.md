# Overview

This is a neural network educational application built as a Binary Digit Trainer. The application allows users to create binary patterns on an 8x8 grid and train a simulated neural network to recognize digits. It features a comprehensive interface with real-time visualization of network training, predictions, metrics, and testing capabilities. The project is built using a modern React frontend with Express.js backend architecture.

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
- **Interactive Binary Grid**: 8x8 clickable grid for creating digit patterns
- **Neural Network Visualization**: Real-time display of network layers and activations
- **Training Controls**: Start/stop training with configurable target labels
- **Metrics Dashboard**: Live accuracy, loss, and training progress tracking
- **Testing Interface**: Sample digit loading and prediction testing
- **Responsive Design**: Mobile-friendly interface with adaptive layouts

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
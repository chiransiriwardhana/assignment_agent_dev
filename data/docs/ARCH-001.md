---
id: ARCH-001
title: Payments platform architecture
department: payments
document_type: architecture
access_level: internal
created_date: 2025-01-15
---
## Overview
payments-core is a Kotlin service cluster fronting PostgreSQL (primary + 2 replicas) and an event bus. card-gateway integrates with an external card processor over mutual TLS.
## Resilience
Pool size 100 per replica (as of January 2025), retries with exponential back-off, no circuit breaker on the gateway client (planned).
## Dependencies
PostgreSQL, Kafka event bus, card processor, HSM-backed key store.

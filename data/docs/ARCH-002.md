---
id: ARCH-002
title: Event bus and messaging standards
department: platform
document_type: architecture
access_level: internal
created_date: 2025-04-01
---
## Overview
All domain events are published to Kafka with schema registry validation. Topics follow the domain.entity.event naming convention.

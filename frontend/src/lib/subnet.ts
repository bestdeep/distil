import subnetConfig from "./subnet-config.json";

export const SUBNET = subnetConfig;
export const TEACHER = subnetConfig.teacher;
export const VALIDATOR = subnetConfig.validator;
export const API_SETTINGS = subnetConfig.api;
export const NETUID = subnetConfig.netuid;
export const SCORE_EPSILON = subnetConfig.validator.epsilon;
export const SCORE_TO_BEAT_FACTOR = 1 - SCORE_EPSILON;
export const API_BASE =
  process.env.API_URL ||
  process.env.NEXT_PUBLIC_API_URL ||
  subnetConfig.api.publicUrl;
export const CLIENT_API_BASE = "";

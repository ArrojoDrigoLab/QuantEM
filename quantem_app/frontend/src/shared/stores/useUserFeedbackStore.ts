import { create } from "zustand";
import type { UserFeedback } from "@/shared/types";

type FeedbackMap = Record<string, UserFeedback>;
type FeedbackBySegmentation = Record<string, FeedbackMap>;

interface UserFeedbackState {
  feedbackBySegmentation: FeedbackBySegmentation;
  upsertFeedback: (segmentationId: string, feedback: UserFeedback) => void;
  upsertFeedbackBatch: (segmentationId: string, feedback: UserFeedback[]) => void;
  removeFeedback: (segmentationId: string, feedbackId: string) => void;
  clearFeedback: (segmentationId: string) => void;
}

export const useUserFeedbackStore = create<UserFeedbackState>((set) => ({
  feedbackBySegmentation: {},
  upsertFeedback: (segmentationId, feedback) =>
    set((state) => {
      const current = state.feedbackBySegmentation[segmentationId] || {};
      return {
        feedbackBySegmentation: {
          ...state.feedbackBySegmentation,
          [segmentationId]: {
            ...current,
            [feedback.id]: feedback,
          },
        },
      };
    }),
  upsertFeedbackBatch: (segmentationId, feedbackItems) =>
    set((state) => {
      const current = state.feedbackBySegmentation[segmentationId] || {};
      const next: FeedbackMap = { ...current };
      for (const feedback of feedbackItems) {
        next[feedback.id] = feedback;
      }
      return {
        feedbackBySegmentation: {
          ...state.feedbackBySegmentation,
          [segmentationId]: next,
        },
      };
    }),
  removeFeedback: (segmentationId, feedbackId) =>
    set((state) => {
      const current = state.feedbackBySegmentation[segmentationId];
      if (!current || !current[feedbackId]) {
        return state;
      }
      const next = { ...current };
      delete next[feedbackId];
      return {
        feedbackBySegmentation: {
          ...state.feedbackBySegmentation,
          [segmentationId]: next,
        },
      };
    }),
  clearFeedback: (segmentationId) =>
    set((state) => ({
      feedbackBySegmentation: {
        ...state.feedbackBySegmentation,
        [segmentationId]: {},
      },
    })),
}));

